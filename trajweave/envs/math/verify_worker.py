"""在线程外运行带 signal 超时的可选数学判分器。"""

import json
import sys

from trajweave.envs.math.c3 import _math_verify_equal_inline


def main() -> None:
    prediction, ground_truth = json.load(sys.stdin)
    print(json.dumps(_math_verify_equal_inline(prediction, ground_truth)))


if __name__ == "__main__":
    main()
