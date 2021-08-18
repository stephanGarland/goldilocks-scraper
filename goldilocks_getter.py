#!/usr/bin/env python3

import argparse
import re
import requests
import yaml
import csv
import kubernetes as k8s
import pandas as pd

from bs4 import BeautifulSoup
from nested_dict import nested_dict

from typing import Dict
from typing import List


def setup_args():
    parser = argparse.ArgumentParser(
        description="""
        This script scrapes current resource requirements and recommendations 
        from Goldilocks for all SOAs in all namespaces, for the provided cluster.
        """
    )

    parser.add_argument(
        "-d",
        "--domain",
        help="Domain where goldilocks is installed, e.g. foo.net",
    )
    
    parser.add_argument(
        "-c",
        "--cluster",
        help="Cluster to get the resources of SOAs in a namespace from",
        default="fast-hedgehog"
    )
    
    parser.add_argument(
        "-n",
        "--namespace",
        help="Namespace (environment) to get resources for",
        default="all"
    )

    parser.add_argument(
        "-f",
        "--file",
        help="Output file to store recommendations in",
    )

    parser.add_argument(
        "-g",
        "--gib",
        help="Use GiB instead of MiB for memory units",
        action="store_true"
    )

    parser.add_argument(
        "-l",
        "--limit",
        help="Stat to use for limits - defaults to max",
        choices=["min", "max", "mean", "median"],
        default="max"
    )

    parser.add_argument(
        "-r",
        "--request",
        help="Stat to use for requests - defaults to median",
        choices=["min", "max", "mean", "median"],
        default="median"
    )

    parser.add_argument(
        "--csv",
        help="Output data in CSV format to easily import it in other tools",
        action="store_true"
    )

    args = parser.parse_args()

    return args


def get_namespaces(cluster: str, namespace: str) -> List[str]:
    """Gets namespaces with Goldilocks.
    Gets namespaces from a given k8s cluster that have Goldilocks enabled.

    Args:
        cluster: k8s cluster name
        namepsace: the namespace to get resource requests for

    Raises:
        Nothing.

    Returns:
        A list of namespaces in the cluster that have Goldilocks enabled.
    """

    client = k8s.client.CoreV1Api(
        api_client=k8s.config.new_client_from_config(
            context=cluster
        )
    )
    
    if namespace != "all":
        return [namespace]
    
    namespaces = client.list_namespace(label_selector="goldilocks.fairwinds.com/enabled")
    filtered_namespaces = []

    for pod in [x.metadata for x in namespaces.items if x.metadata.labels]:
        filtered_namespaces.append(pod.name)

    return filtered_namespaces

def get_current_resources(
    cluster: str,
    namespace: str,
) -> Dict[str, k8s.client.V1ResourceRequirements]:
    """Gets current resource requirements for SOAs in the provided namespace and cluster.
    Gets current resource requirements for SOAs in the provided namespace and cluster.

    Args:
        cluster: k8s cluster name
        namepsace: the namespace to get resource requests for

    Raises:
        Nothing.

    Returns:
        A dictionary with cluster/namespace/SOA key and the SOAs' resource requirements.
    """

    client = k8s.client.AppsV1Api(
        api_client=k8s.config.new_client_from_config(
            context=cluster
        )
    )

    mega_dict = nested_dict()
    namespaces = get_namespaces(cluster, namespace)

    print(f"Getting current requirements for SOAs in {cluster}/{namespace}")

    deployments = client.list_namespaced_deployment(namespace=namespace)
    for deployment in deployments.items:
        for container in deployment.spec.template.spec.containers:
            mega_dict[cluster][namespace][deployment.metadata.name] = container.resources

    return mega_dict


def get_html(
    cluster: str,
    domain: str,
    namespace: str
) -> bytes:
    """Gets HTML output for a namespace from Goldilocks.
    Uses requests to get an HTML page from Goldilocks for a specified namespace.

    Args:
        cluster: The k8s cluster.
        domain: The domain where Goldilocks is installed.
        namespace: The requested namespace.

    Raises:
        SystemExit in response to a Requests error.

    Returns:
        Bytes from Goldilocks.
    """

    # Test multiple hostname from domain separators.
    # The endpoint should normally be a FQDN (e.g. use a dot).
    separators = ["-", "."]

    for index, sep in enumerate(separators):
        try:
            endpoint = f"https://goldilocks.{cluster}{sep}{domain}/dashboard/{namespace}"

            print(f"Attempt connecting to {endpoint}")
            request = requests.get(endpoint)
            print(f"Successfully connected to {endpoint}")

            return request.content
        except requests.exceptions.ConnectionError:
            if index < len(separators):
                print(f"Failed connecting to {endpoint}")
            else:
                print(f"Exception raised when connecting to {endpoint}")
                raise SystemExit


def make_soup(
        html: bytes,
        qos: str = "Burstable"
) -> Dict[str, str]:
    """Uses BeautifulSoup to return VPA recommendations.
    Uses BeautifulSoup to parse a truly horrifying series of DOM navigation
    on raw HTML to retrieve Goldilock's recommendations for a given QoS.

    Args:
        html: bytes from Goldilocks.
        qos: Burstable or Guaranteed - defaults to Burstable.

    Raises:
        AtrributeError: If the DOM path is not found.

    Returns:
        A Dict[str, str] containing Goldilock's recommendations.
    """

    soup = BeautifulSoup(html, "html.parser")
    soup_dict = {}
    for ele in soup.find_all("code"):
        try:
            if qos in ele.parent.parent.parent.h5.text:
                # I'm so sorry.
                new_ele_name = ele.parent.parent.parent.parent.parent.parent.summary.h3.text.strip().split("\n")[1].strip()
                soup_dict[new_ele_name] = ele.text
        except AttributeError:
            print("Unable to navigate the DOM - it may have changed.")
            raise SystemExit

    return soup_dict


def convert_human_readable_to_bytes(human_bytes: str) -> int:
    """Converts human-readable numbers to bytes.
    Converts human-readable numbers like MB, GiB to bytes.

    Args:
        human_bytes: An input string in human-readable format.

    Raises:
        Nothing.

    Returns:
        An int of bytes.
    """
    convert_map = {
        "k": 10**3,
        "m": 10**6,
        "g": 10**9,
        "t": 10**12,
        "ki": 2**10,
        "mi": 2**20,
        "gi": 2**30,
        "ti": 2**40
    }

    human_bytes_tuple = re.search("([0-9]+)([A-z]+)", human_bytes).groups()

    return int(human_bytes_tuple[0]) * convert_map[human_bytes_tuple[1].lower()]


def convert_bytes_to_human_readable(machine_bytes: int, gib: bool = False) -> str:
    """Converts bytes to mebibytes or gibibytes.
    Converts bytes (e.g. 1073741824) to human-readable (e.g. 1 [GiB] - unit not included)

    Args:
        gib: Use GiB (True) or MiB (False) for units.
        machine_bytes: An input int of bytes.

    Raises:
        Nothing.

    Returns:
        A string of human-readable bytes, e.g. 1 [GiB].
    """

    if gib:
        multiplicand = 30
    else:
        multiplicand = 20

    return f"{int(round(machine_bytes, 2)/2**multiplicand)}"


def make_useful_dict(
    mega_dict: nested_dict,
    source: str
) -> nested_dict:
    """Makes a nested_dict() with useful information.
    Makes a nested_dict() with limits and requests for CPU and memory,
    along with cluster and namespace information.

    Args:
        mega_dict: A nested_dict.
        source: Source of data (html or k8s).

    Raises:
        Nothing.

    Returns:
        A nested_dict.
    """

    soa_recs = nested_dict()
    if source == "html":
        # First, flatten out a given cluster's items
        for k, v in mega_dict.items_flat():
            # Example line for a given SOA:
            # ['resources:', '  requests:', '    cpu: 108m', '    memory: 5815M', '  limits:', '    cpu: 191m', '    memory: 7134M']
            line = v.split("\n")

            # Strip out whitespace and split values out, converting to bytes and dropping millicore unit
            request_cpu = line[2].split(":")[1].replace(" ", "").replace("m", "")
            request_mem = line[3].split(":")[1].replace(" ", "")
            request_mem = convert_human_readable_to_bytes(request_mem)
            limit_cpu = line[5].split(":")[1].replace(" ", "").replace("m", "")
            limit_mem = line[6].split(":")[1].replace(" ", "")
            limit_mem = convert_human_readable_to_bytes(limit_mem)

            # Fill new nested_dict with values
            soa_recs[k[0]]["requests"][k[1]][k[2]]["cpu"] = int(request_cpu)
            soa_recs[k[0]]["requests"][k[1]][k[2]]["memory"] = int(request_mem)
            soa_recs[k[0]]["limits"][k[1]][k[2]]["cpu"] = int(limit_cpu)
            soa_recs[k[0]]["limits"][k[1]][k[2]]["memory"] = int(limit_mem)
    if source == "k8s":
        # First, flatten out a given cluster's items
        for k, v in mega_dict.items_flat():
            # Example line for a given SOA:
            # ['resources:', '  requests:', '    cpu: 108m', '    memory: 5815M', '  limits:', '    cpu: 191m', '    memory: 7134M']

            # Strip out whitespace and split values out, converting to bytes and dropping millicore unit
            try:
                request_cpu = v.requests["cpu"]
            except KeyError:
                request_cpu = "0"
            try:
                request_mem = v.requests["memory"]
            except KeyError:
                request_mem = "0"

            request_mem = convert_human_readable_to_bytes(request_mem)

            try:
                limit_cpu = v.limits["cpu"]
            except KeyError:
                limit_cpu = "0"
            try:
                limit_mem = v.limits["memory"]
            except KeyError:
                limit_mem = "0"
            limit_mem = convert_human_readable_to_bytes(limit_mem)

            # Fill new nested_dict with values
            soa_recs[k[0]]["requests"][k[1]][k[2]]["cpu"] = int(request_cpu.replace('m', ''))
            soa_recs[k[0]]["requests"][k[1]][k[2]]["memory"] = int(request_mem)
            soa_recs[k[0]]["limits"][k[1]][k[2]]["cpu"] = int(limit_cpu.replace('m', ''))
            soa_recs[k[0]]["limits"][k[1]][k[2]]["memory"] = int(limit_mem)

    return soa_recs
        

def make_dataframe(
    recs: nested_dict,
    limit_or_rec: str,
    cluster: str,
    gib: bool
) -> tuple:
    """Makes a dataframe for each resource type and and fills it out.
    Makes a Pandas dataframe for CPU and memory, and fills it with usable data from Goldilocks.

    Args:
        recs: A nested_dict containing Goldilocks' recommendations.
        limit_or_rec: {requests, limits} - k8s resource recommendation type.
        cluster: cluster which hosts the environment.
        gib: From args.gib - indicates whether to use Gi[B] or Mi[B].


    Raises:
        SystemExit if unable to find any recommendations.

    Returns:
        A tuple of two dataframes.
    """

    cpu_values = []
    memory_values = []
    cpu_columns = ["namespace", "soa", "millicores"]
    memory_columns = ["namespace", "soa", "size"]

    for k, v in recs[cluster][limit_or_rec].items_flat():
        if "memory" in k:
            memory_values.append((k[0], k[1], convert_bytes_to_human_readable(v, gib)))
        elif "cpu" in k:
            cpu_values.append((k[0], k[1], v))
        else:
            print("Error finding recommendations - can you connect to the cluster?")
            raise SystemExit

    cpu_df = pd.DataFrame(
        cpu_values,
        columns=cpu_columns,
        dtype=int
    )
    memory_df = pd.DataFrame(
        memory_values,
        columns=memory_columns,
        dtype=int
    )

    return cpu_df, memory_df


def make_aggregate_dataframes(
    cpu_requests_df: pd.DataFrame,
    memory_requests_df: pd.DataFrame,
    cpu_limits_df: pd.DataFrame,
    memory_limits_df: pd.DataFrame,
) -> tuple:
    """Makes aggregate dataframes.
    Makes Pandas dataframes and fills them with aggregate info for implementation.

    Args:
        recs: A nested_dict containing Goldilocks' recommendations.

    Raises:
        Nothing.

    Returns:
        A tuple of dataframes.
    """

    stats = ["min", "max", "mean", "median"]
    # CPU is already in millicores, no need for more precision
    cpu_limits_df = cpu_limits_df.groupby(["soa"]).agg({"millicores": stats}).round(0)
    cpu_requests_df = cpu_requests_df.groupby(["soa"]).agg({"millicores": stats}).round(0)
    memory_requests_df = memory_requests_df.groupby(["soa"]).agg({"size": stats}).round(2)
    memory_limits_df = memory_limits_df.groupby(["soa"]).agg({"size": stats}).round(2)

    return cpu_requests_df, cpu_limits_df, memory_requests_df, memory_limits_df


def get_recs(cluster: str, domain: str, namespace: str) -> dict:
    """Calls most of the functions to get recommendations.
    Calls other functions to generate a nested_dict containing all recommendations.

    Args:
        cluster: The cluster to get resources from.
        domain: The domain name where Goldilocks is installed.
        namespace: The namespace to get resource requests for.

    Raises:
        Nothing.

    Returns:
        A tuple with the cluster list and a nested_dict.
    """

    mega_dict = nested_dict()

    print(f"Getting recommendations for SOAs in {cluster}/{namespace}")

    html = get_html(cluster, domain, namespace)
    soup_dict = make_soup(html)
    for soa, recs in soup_dict.items():
        mega_dict[cluster][namespace][soa] = recs

    return mega_dict


def make_resource_dict(
    cpu_req_tuple_df: tuple,
    mem_req_tuple_df: tuple,
    cpu_lim_tuple_df: tuple,
    mem_lim_tuple_df: tuple,
    soa: str,
    lim_stat: str,
    req_stat: str,
    gib: bool
) -> dict:
    """Makes a dict to be converted to YAML.
    Gets {stat} value from the dataframe (e.g. max),
    and creates a dict to be converted to YAML.

    Args:
        cpu_req_df: A dataframe containing CPU requests.
        mem_req_df: A dataframe containing memory requests.
        cpu_lim_df: A dataframe containing CPU limits.
        mem_lim_df: A dataframe containing memory limits.
        soa: A str of the SOA's name to select.
        {lim,req}stat: A str of {min, max, mean, median} to select that stat.
        gib: From args.gib - indicates whether to use Gi[B] or Mi[B].

    Raises:
        System Exit if KeyError, indicative of a problem retreiving or parsing the data.

    Returns:
        A dict of resources.
    """

    if args.gib:
        unit = "Gi"
    else:
        unit = "Mi"

    cpu_req_df = cpu_req_tuple_df[0]
    mem_req_df = mem_req_tuple_df[0]
    cpu_lim_df = cpu_lim_tuple_df[0]
    mem_lim_df = mem_lim_tuple_df[0]
    rec_cpu_req_df = cpu_req_tuple_df[1]
    rec_mem_req_df = mem_req_tuple_df[1]
    rec_cpu_lim_df = cpu_lim_tuple_df[1]
    rec_mem_lim_df = mem_lim_tuple_df[1]

    try:
        soa_dict = nested_dict()
        soa_dict["resources"]["current"]["limits"]["cpu"] = f"{cpu_lim_df[('millicores'), lim_stat][soa]}m"
        soa_dict["resources"]["current"]["limits"]["memory"] = f"{mem_lim_df[('size', lim_stat)][soa]}{unit}"
        soa_dict["resources"]["current"]["requests"]["cpu"] = f"{cpu_req_df[('millicores', req_stat)][soa]}m"
        soa_dict["resources"]["current"]["requests"]["memory"] = f"{mem_req_df[('size', req_stat)][soa]}{unit}"
        soa_dict["resources"]["recommendations"]["limits"]["cpu"] = f"{rec_cpu_lim_df[('millicores'), lim_stat][soa]}m"
        soa_dict["resources"]["recommendations"]["limits"]["memory"] = f"{rec_mem_lim_df[('size', lim_stat)][soa]}{unit}"
        soa_dict["resources"]["recommendations"]["requests"]["cpu"] = f"{rec_cpu_req_df[('millicores', req_stat)][soa]}m"
        soa_dict["resources"]["recommendations"]["requests"]["memory"] = f"{rec_mem_req_df[('size', req_stat)][soa]}{unit}"
    except KeyError as e:
       print(f"Error finding {soa} - {e}")

    return soa_dict


def make_yaml_resource_block(namespace: str, soa: dict) -> str:
    """Makes a formatted yaml resource block.
    Takes an input dict generated from dataframes,
    and formats it as below for inclusion in a Helm chart.

    resources:
      current:
        limits:
          cpu: 1151m
          memory: 4999Mi
        requests:
          cpu: 812m
          memory: 4506Mi
      recommendations:
        limits:
          cpu: 45m
          memory: 300Mi
        requests:
          cpu: 15m
          memory: 150Mi

    Args:
        soa: A dict containing current and recommended CPU & Memory values.

    Raises:
        Nothing.

    Returns:
        A formatted string in YAML format.
    """

    return f"\n{yaml.dump({namespace: soa})}\n"


def build_resources_dict(
    soas: tuple,
    cpu_agg_req_df: tuple,
    mem_agg_req_df: tuple,
    cpu_agg_lim_df: tuple,
    mem_agg_lim_df: tuple,
    gib: bool,
    stat_lim: str = "max",
    stat_req: str = "median"
) -> dict:
    """Creates a single string of recommendations per SOA.
    Creates a single formatted string - not necessarily to YAML spec,
    since there are multiple SOAs on the same page - for manual input.

    Args:
        soas: The index from an aggregate dataframe, which is a list of SOAs.
        {cpu,mem}_agg_{req,lim}_df: Dataframes containing aggregates for resources.
        gib: From args.gib - indicates whether to use Gi[B] or Mi[B].
        stat_{lim,req}: A str of {min, max, mean, median} to select that stat.


    Raises:
        Nothing.

    Returns:
        A dictionary of all current and recommended values for resources.
    """

    soa_recs = {}

    for soa in soas[0]:
        soa_nested_dict = make_resource_dict(
            cpu_agg_req_df,
            mem_agg_req_df,
            cpu_agg_lim_df,
            mem_agg_lim_df,
            soa,
            stat_lim,
            stat_req,
            gib,
        )
        soa_rec = soa_nested_dict.to_dict()
        soa_recs[soa] = soa_rec

    return soa_recs


def export_csv(filename: str, namespace, resources: dict):
    """Exports current, recommended and the difference in resources in a CSV file.
    Exports current, recommended by Goldilocks and the difference in resources in
    a CSV file that can easily be imported into spreadsheets or other data analysis
    tools to graph the data.

    Args:
        filename: The filename that will be used to create the CSV file.
        resources: The current and recommended resources by Goldilocks.

    Raises:
        Nothing.

    Returns:
        Nothing.
    """

    if args.gib:
        unit = "Gi"
    else:
        unit = "Mi"

    headers = [
        "namespace",
        "soa",
        "Current [CPU] (m)",
        "Recommendations [CPU] (m)",
        "Difference [CPU] (m)",
        f"Current [Memory] ({unit})",
        f"Recommendations [Memory] ({unit})",
        f"Difference [Memory] ({unit})"
    ]

    with open(f"{filename}.csv", 'a') as outp:
        dw = csv.DictWriter(outp, delimiter=',', fieldnames=headers)
        dw.writeheader()

        for k, v in resources.items():
            try:
                cur_cpu = v['resources']['current']['requests']['cpu'].strip("m")
                rec_cpu = v['resources']['recommendations']['requests']['cpu'].strip("m")
                diff_cpu = rec_cpu - cur_cpu
                cur_mem = v['resources']['current']['requests']['memory'].strip("Mi")
                rec_mem = v['resources']['recommendations']['requests']['memory'].strip("Mi")
                diff_memory = rec_memory - cur_memory
            except:
                rec_cpu = 0
                cur_cpu = 0
                diff_cpu = 0
                rec_mem = 0
                cur_mem = 0
                diff_memory = 0

            outp.write(f"{namespace},{k},{cur_cpu},{rec_cpu},{diff_cpu},{cur_mem},{rec_mem},{diff_memory}\n")



if __name__ == "__main__":
    args = setup_args()

    print("Initializing...")
    namespaces = get_namespaces(args.cluster, args.namespace)

    for namespace in namespaces:
        mega_dict = get_recs(args.cluster, args.domain, namespace)
        useful_mega_dict = make_useful_dict(mega_dict, "html")

        rec_cpu_lim_df, rec_mem_lim_df = make_dataframe(
            useful_mega_dict,
            "limits",
            args.cluster,
            args.gib
        )

        rec_cpu_req_df, rec_mem_req_df = make_dataframe(
            useful_mega_dict,
            "requests",
            args.cluster,
            args.gib
        )

        rec_cpu_agg_req_df, rec_cpu_agg_lim_df, rec_mem_agg_req_df, rec_mem_agg_lim_df = make_aggregate_dataframes(
            rec_cpu_req_df,
            rec_mem_req_df,
            rec_cpu_lim_df,
            rec_mem_lim_df
        )

        mega_dict = get_current_resources(args.cluster, namespace)
        useful_mega_dict = make_useful_dict(mega_dict, "k8s")

        cpu_lim_df, mem_lim_df = make_dataframe(
            useful_mega_dict,
            "limits",
            args.cluster,
            args.gib
        )
        cpu_req_df, mem_req_df = make_dataframe(
            useful_mega_dict,
            "requests",
            args.cluster,
            args.gib
        )

        cpu_agg_req_df, cpu_agg_lim_df, mem_agg_req_df, mem_agg_lim_df = make_aggregate_dataframes(
            cpu_req_df,
            mem_req_df,
            cpu_lim_df,
            mem_lim_df
        )

        resources = build_resources_dict(
            (cpu_agg_req_df.index,rec_cpu_agg_req_df.index),
            (cpu_agg_req_df, rec_cpu_agg_req_df),
            (mem_agg_req_df,rec_mem_agg_req_df),
            (cpu_agg_lim_df,rec_cpu_agg_lim_df),
            (mem_agg_lim_df,rec_mem_agg_lim_df),
            args.gib,
            args.limit,
            args.request
        )
        yaml_dump = make_yaml_resource_block(namespace, resources)

        filename = args.file
        if filename is None:
            filename = namespace

        with open(f"{filename}.yaml", "a") as f:
            f.write(yaml_dump)

        if args.csv:
            export_csv(filename, namespace, resources)

