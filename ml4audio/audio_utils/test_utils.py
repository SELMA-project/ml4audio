import os
import shutil

from misc_utils.prefix_suffix import BASE_PATHES, PrefixSuffix

TEST_RESOURCES = "tests/resources"
BASE_PATHES["test_resources"] = TEST_RESOURCES


def get_test_cache_base():
    cache_base = PrefixSuffix("test_resources", "cache")
    if (
        os.path.isdir(str(cache_base))
        and not os.environ.get("DONT_REMOVE_TEST_CACHE", "False") != "False"
    ):
        shutil.rmtree(str(cache_base))
    os.makedirs(str(cache_base), exist_ok=True)
    return cache_base


def get_test_vocab():
    return f"""<pad>
<s>
</s>
<unk>
|
E
T
A
O
N
I
H
S
R
D
L
U
M
W
C
F
G
Y
P
B
V
K
'
X
J
Q
Z""".split(
        "\n"
    )
