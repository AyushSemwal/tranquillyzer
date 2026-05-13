from typing import Optional


def split_bam_wrap(
    input_bam: str,
    out_dir: str,
    bucket_threads: Optional[int] = None,
    merge_threads: Optional[int] = None,
    nbuckets: int = 256,
    tag: str = "CB",
    max_open_cb_writers: int = 128,
    filter_secondary: bool = False,
    filter_supplementary: bool = False,
    filter_unmapped: bool = True,
    filter_duplicates: bool = True,
    min_mapq: Optional[int] = None,
    keep_tmp: bool = False,
    index_outputs: bool = True,
    prefer_csi_index: bool = False,
):
    """Split a BAM file into per-cell-barcode BAM files."""
    import logging
    import os
    import resource
    import time

    from scripts.split_bam_file import split_bam_file

    logger = logging.getLogger(__name__)

    start = time.time()

    split_bam_file(
        input_bam,
        out_dir,
        bucket_threads=bucket_threads,
        merge_threads=merge_threads,
        nbuckets=nbuckets,
        tag=tag,
        max_open_cb_writers=max_open_cb_writers,
        filter_secondary=filter_secondary,
        filter_supplementary=filter_supplementary,
        filter_unmapped=filter_unmapped,
        filter_duplicates=filter_duplicates,
        min_mapq=min_mapq,
        keep_tmp=keep_tmp,
        index_outputs=index_outputs,
        prefer_csi_index=prefer_csi_index,
    )

    usage_self = resource.getrusage(resource.RUSAGE_SELF)
    usage_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_kb = usage_self.ru_maxrss + usage_children.ru_maxrss
    max_rss_mb = max_rss_kb / 1024 if os.uname().sysname == "Linux" else max_rss_kb
    logger.info(f"Peak memory usage: {max_rss_mb:.2f} MB")
    logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
