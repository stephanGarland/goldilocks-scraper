# Goldilocks Recommendation Scraper

### Acknowledgement
This would be an utterly useless tool without [Goldilocks.](https://goldilocks.docs.fairwinds.com/)

### Why does this exist?
While Goldilocks [does have a CLI](https://goldilocks.docs.fairwinds.com/advanced/#cli-usage-not-recommended) that can output JSON, it doesn't clean up after itself, and is scoped to a single namespace.

### Couldn't you have just run that in a for loop with kubectl get namespaces?

Go away. 

### Limitations
This does not change your k8s resource limits or requests, nor does it edit files for you. Honestly, that's probably a good thing. If you want to extend this to calling kubectl apply, be my guest, and be sure to post your prod disaster story later.

### Prerequisites

 - Python 3.6+ for Typing and f-strings
 - Goldilocks installed in at least one cluster
 - A .kubeconfig file or some other method of authenticating with the cluster.
 - Packages from requirements.txt
	 - Note that you may have to [change the kubernetes library version](https://github.com/kubernetes-client/python#compatibility) depending on your cluster's version

### Usage

    usage: goldilocks_getter.py [-h] -d DOMAIN -f FILE [-m] \
           [-l {min,max,mean,median}] [-r {min,max,mean,median}] [-t]

    This script scrapes recommendations from Goldilocks for all SOAs
    in all namespaces, in all clusters available in the user's .kubeconfig.

    optional arguments:
      -h, --help            show this help message and exit
      -d DOMAIN, --domain DOMAIN
                            domain where goldilocks is installed, 
                            e.g. goldilocks.foo.net
      -f FILE, --file FILE  Output file to store recommendations in
      -m, --mib             Use MiB instead of GiB for memory units
      -l {min,max,mean,median}, --limit {min,max,mean,median}
                            Stat to use for limits - defaults to max
      -r {min,max,mean,median}, --request {min,max,mean,median}
                            Stat to use for requests - defaults to median
      -t, --test            Only collect information from the first 
                            cluster to speed up testing

### Example output

    soa_1:
      resources:
        limits:
          cpu: 151.0m
          memory: 2.13Gi
        requests:
          cpu: 62.0m
          memory: 2.01Gi
    soa_2:
      resources:
        limits:
          cpu: 25.0m
          memory: 0.24Gi
        requests:
          cpu: 25.0m
          memory: 0.24Gi
