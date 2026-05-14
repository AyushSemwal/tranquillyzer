import logging

logger = logging.getLogger(__name__)


def read_length_distr_wrap(output_dir):
    """Plot read-length distribution from preprocessed Parquet files."""
    import os
    import time
    import resource
    from scripts.plot_read_len_distr import plot_read_len_distr

    start = time.time()
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    logger.info("Generating the read-length distribution plot")
    plot_read_len_distr(output_dir + "/full_length_pp_fa", output_dir + "/plots")
    logger.info("Finished generating the read-length distribution plot")

    usage_self = resource.getrusage(resource.RUSAGE_SELF)
    usage_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_kb = usage_self.ru_maxrss + usage_children.ru_maxrss
    max_rss_mb = max_rss_kb / 1024 if os.uname().sysname == "Linux" else max_rss_kb
    logger.info(f"Peak memory usage during alignment: {max_rss_mb:.2f} MB")
    logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
