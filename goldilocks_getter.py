#!/usr/bin/env python3

import argparse
import re
import requests
from typing import Dict
from typing import List
import yaml

from bs4 import BeautifulSoup
import kubernetes as k8s
from nested_dict import nested_dict
import pandas as pd


def setup_args():
    parser = argparse.ArgumentParser(
        description="""
        This script scrapes recommendations from Goldilocks for all SOAs
        in all namespaces, in all clusters available in the user's .kubeconfig.
        """
    )

    parser.add_argument(
        "-d",
        "--domain",
        help="domain where goldilocks is installed, e.g. goldilocks.foo.net",
        required=True
    )

    parser.add_argument(
        "-f",
        "--file",
        help="Output file to store recommendations in",
        required=True
    )

    parser.add_argument(
        "-m",
        "--mib",
        help="Use MiB instead of GiB for memory units",
        action="store_true"
    )

    parser.add_argument(
        "-t",
        "--test",
        help="Only collect information from the first cluster to speed up testing",
        action="store_true"
    )

    args = parser.parse_args()

    return args


def get_clusters(env: str, test: bool = False) -> List[str]:
    """Gets prod k8s clusters.
    Uses the user's kubeconfig file to get all clusters available, and then
    filters to the user's desired environment. If the _ in the call to
    list_kube_config_contexts() is replaced with a variable, the active context
    will also be returned.

    Args:
        env: A filter to be used for startswith() - {prod, stage, test}.
        test: Only returns the first cluster.

    Raises:
        Nothing.

    Returns:
        A list of prod k8s clusters.
    """

    contexts, _ = k8s.config.list_kube_config_contexts()
    filtered_contexts = [x["name"] for x in contexts if x["name"].startswith(env)]
    if test:
        return filtered_contexts[:1]
    return [x["name"] for x in contexts if x["name"].startswith(env)]


def get_namespaces(cluster: str) -> List[str]:
    """Gets namespaces with Goldilocks.
    Gets namespaces from a given k8s cluster that have Goldilocks enabled.

    Args:
        cluster: k8s cluster name

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
    namespaces = client.list_namespace()
    filtered_namespaces = []

    for pod in [x.metadata for x in namespaces.items if x.metadata.labels]:
        if pod.labels["goldilocks.fairwinds.com/enabled"]:
            filtered_namespaces.append(pod.name)

    return filtered_namespaces


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

    cluster_name = "-".join(cluster.split("-")[:-1])
    try:
        endpoint = f"https://goldilocks-{cluster_name}.{domain}/dashboard/{namespace}"
        request = requests.get(endpoint)
    except requests.exceptions.ConnectionError:
        print(f"Exception raised when connecting to {endpoint}")
        raise SystemExit

    return request.content


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


def convert_bytes_to_human_readable(machine_bytes: int, mib: bool = False) -> str:
    """Converts bytes to mebibytes or gibibytes.
    Converts bytes (e.g. 1073741824) to human-readable (e.g. 1 [GiB] - unit not included)

    Args:
        mib: Use MiB (True) or GiB (False) for units.
        machine_bytes: An input int of bytes.

    Raises:
        Nothing.

    Returns:
        A string of human-readable bytes, e.g. 1 [GiB].
    """

    if mib:
        multiplicand = 20
    else:
        multiplicand = 30

    return f"{round(machine_bytes, 2)/2**multiplicand}"


def make_useful_dict(mega_dict: nested_dict) -> nested_dict:
    """Makes a nested_dict() with useful information.
    Makes a nested_dict() with limits and requests for CPU and memory,
    along with cluster and namespace information.

    Args:
        mega_dict: A nested_dict.

    Raises:
        Nothing.

    Returns:
        A nested_dict.
    """

    soa_recs = nested_dict()
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

    return soa_recs


def make_dataframe(
    recs: nested_dict,
    limit_or_rec: str,
    cluster_list: list,
    mib: bool
) -> tuple:
    """Makes a dataframe for each resource type and and fills it out.
    Makes a Pandas dataframe for CPU and memory, and fills it with usable data from Goldilocks.

    Args:
        recs: A nested_dict containing Goldilocks' recommendations.
        limit_or_rec: {requests, limits} - k8s resource recommendation type.
        mib: From args.mib - indicates whether to use Gi[B] or Mi[B].


    Raises:
        SystemExit if unable to find any recommendations.

    Returns:
        A tuple of two dataframes.
    """

    cpu_values = []
    memory_values = []
    cpu_columns = ["namespace", "soa", "millicores"]
    memory_columns = ["namespace", "soa", "size"]

    for cluster in cluster_list:
        for k, v in recs[cluster][limit_or_rec].items_flat():
            if "memory" in k:
                memory_values.append((k[0], k[1], convert_bytes_to_human_readable(v, mib)))
            elif "cpu" in k:
                cpu_values.append((k[0], k[1], int(v)))
            else:
                print("Error finding recommendations - can you connect to the cluster?")
                raise SystemExit

    cpu_df = pd.DataFrame(cpu_values, columns=cpu_columns, dtype=float)
    memory_df = pd.DataFrame(
        memory_values, columns=memory_columns, dtype=float)

    return cpu_df, memory_df


def make_aggregate_dataframes(
    cpu_requests_df: pd.DataFrame,
    memory_requests_df: pd.DataFrame,
    cpu_limits_df: pd.DataFrame,
    memory_limits_df: pd.DataFrame
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


def get_recs(domain: str, test: bool = False) -> tuple:
    """Calls most of the functions to get recommendations.
    Calls other functions to generate a nested_dict containing all recommendations.

    Args:
        domain: The domain name where Goldilocks is installed.
        test: Passed to get_clusters() to only return the first cluster.

    Raises:
        Nothing.

    Returns:
        A tuple with the cluster list and a nested_dict.
    """
    cluster_namespaces = {}
    mega_dict = nested_dict()
    clusters = get_clusters("prod", test)

    for cluster in clusters:
        cluster_namespaces[cluster] = get_namespaces(cluster)

    for cluster_name, namespace_list in cluster_namespaces.items():
        print(f"Getting data from {cluster_name}")
        for namespace in namespace_list:
            html = get_html(cluster_name, domain, namespace)
            soup_dict = make_soup(html)
            for soa, recs in soup_dict.items():
                mega_dict[cluster_name][namespace][soa] = recs

    return clusters, mega_dict


def make_resource_dict(
    cpu_req_df: pd.DataFrame,
    mem_req_df: pd.DataFrame,
    cpu_lim_df: pd.DataFrame,
    mem_lim_df: pd.DataFrame,
    soa: str,
    lim_stat: str,
    req_stat: str,
    mib: bool
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
        mib: From args.mib - indicates whether to use Gi[B] or Mi[B].

    Raises:
        System Exit if KeyError, indicative of a problem retreiving or parsing the data.

    Returns:
        A dict of resources.
    """

    if args.mib:
        unit = "Mi"
    else:
        unit = "Gi"

    try:
        soa_dict = nested_dict()
        soa_dict["resources"]["limits"]["cpu"] = f"{cpu_lim_df[('millicores'), lim_stat][soa]}m"
        soa_dict["resources"]["limits"]["memory"] = f"{mem_lim_df[('size', lim_stat)][soa]}{unit}"
        soa_dict["resources"]["requests"]["cpu"] = f"{cpu_req_df[('millicores', req_stat)][soa]}m"
        soa_dict["resources"]["requests"]["memory"] = f"{mem_req_df[('size', req_stat)][soa]}{unit}"
    except KeyError as e:
        print(f"Error finding {soa} - {e}")
        raise SystemExit

    return soa_dict.to_dict()


def make_yaml_resource_block(soa: dict) -> str:
    """Makes a formatted yaml resource block.
    Takes an input dict generated from dataframes,
    and formats it as below for inclusion in a Helm chart.

    resources:
      requests:
        cpu: 812m
        memory: 4506M
      limits:
        cpu: 1151m
        memory: 4999M

    Args:
        soa: A dict containing CPU and memory values.

    Raises:
        Nothing.

    Returns:
        A formatted string in YAML format.
    """

    return f"\n{yaml.dump(soa)}\n"


def make_yaml_dump(
    soas: pd.core.indexes.base.Index,
    cpu_agg_req_df: pd.DataFrame,
    mem_agg_req_df: pd.DataFrame,
    cpu_agg_lim_df: pd.DataFrame,
    mem_agg_lim_df: pd.DataFrame,
    mib: bool
) -> str:
    """Creates a single string of recommendations per SOA.
    Creates a single formatted string - not necessarily to YAML spec,
    since there are multiple SOAs on the same page - for manual input.

    Args:
        soas: The index from an aggregate dataframe, which is a list of SOAs.
        {cpu,mem}_agg_{req,lim}_df: Dataframes containing aggregates for resources.
        mib: From args.mib - indicates whether to use Gi[B] or Mi[B].


    Raises:
        Nothing.

    Returns:
        A formatted string of all recommendations.
    """
    STAT_FOR_LIMIT = "max"
    STAT_FOR_REQUEST = "median"
    soa_recs = {}

    for soa in soas:
        soa_rec = make_resource_dict(
            cpu_agg_req_df,
            mem_agg_req_df,
            cpu_agg_lim_df,
            mem_agg_lim_df,
            soa,
            STAT_FOR_LIMIT,
            STAT_FOR_REQUEST,
            mib
        )
        soa_recs[soa] = soa_rec

    return make_yaml_resource_block(soa_recs)


if __name__ == "__main__":
    args = setup_args()

    print("Initializing...")
    clusters, mega_dict = get_recs(args.domain, args.test)
    useful_mega_dict = make_useful_dict(mega_dict)

    cpu_lim_df, mem_lim_df = make_dataframe(
        useful_mega_dict,
        "limits",
        clusters,
        args.mib
    )
    cpu_req_df, mem_req_df = make_dataframe(
        useful_mega_dict,
        "requests",
        clusters,
        args.mib
    )
    cpu_agg_req_df, cpu_agg_lim_df, mem_agg_req_df, mem_agg_lim_df = make_aggregate_dataframes(
        cpu_req_df,
        mem_req_df,
        cpu_lim_df,
        mem_lim_df
    )

    recs = make_yaml_dump(
        cpu_agg_req_df.index,
        cpu_agg_req_df,
        mem_agg_req_df,
        cpu_agg_lim_df,
        mem_agg_lim_df,
        args.mib
    )

    with open(args.file, "w") as f:
        f.write(recs)
