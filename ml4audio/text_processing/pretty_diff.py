from typing import Optional

from beartype import beartype

from ml4audio.text_processing.smith_waterman_alignment import align_split


@beartype
def smithwaterman_aligned_icdiff(
    ref: str,
    hyp: str,
    split_len_a=70,
    ref_header: Optional[str] = "ref",
    hyp_header: Optional[str] = "hyp",
) -> str:
    import icdiff

    refs, hyps = align_split(ref, hyp, split_len_a=split_len_a, debug=False)
    cd = icdiff.ConsoleDiff(cols=2 * split_len_a + 20)

    diff_line = "\n".join(
        cd.make_table(
            refs,
            hyps,
            ref_header,
            hyp_header,
        )
    )
    return diff_line


if __name__ == "__main__":
    ref = "NOT HAVING THE COURAGE OR THE INDUSTRY OF OUR NEIGHBOUR WHO WORKS LIKE A BUSY BEE IN THE WORLD OF MEN AND BOOKS SEARCHING WITH THE SWEAT OF HIS BROW FOR THE REAL BREAD OF LIFE WETTING THE OPEN PAGE BEFORE HIM WITH HIS TEARS PUSHING INTO THE WE HOURS OF THE NIGHT HIS QUEST ANIMATED BY THE FAIREST OF ALL LOVES THE LOVE OF TRUTH WE EASE OUR OWN INDOLENT CONSCIENCE BY CALLING HIM NAMES"
    hyp = "NOT HAVING THE COURAGE OR THE INDUSTRY OF OUR NEIGHBOUR WHO WORKS LIKE A BUSY BEE IN THE WORLD OF MEN AN BOOKS SEARCHING WITH THE SWEAT OF HIS BROW FOR THE REAL BREAD OF LIFE WET IN THE OPEN PAGE BAFORE HIM WITH HIS TEARS PUSHING INTO THE WEE HOURS OF THE NIGHT HIS QUEST AND BY THE FAIREST OF ALL LOVES THE LOVE OF TRUTH WE EASE OUR OWN INDOLENT CONSCIENCE BY CALLING HIM NAMES"

    print(smithwaterman_aligned_icdiff(ref, hyp, ref_header=None, hyp_header=None))
