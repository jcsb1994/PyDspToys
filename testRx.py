'''
simulations rxRegex

'''

import re
import sys
from dataclasses import dataclass
from typing import Callable, Optional, List

@dataclass
class RxCmd:
    cmd : str
    rxRegex : str
    requiredRespLen = 0
    response = None
    expectedResponse : List = None
    def __init__(self, cmd, rxRegex, **kwds):
        compiled_pattern = re.compile(rxRegex)
        self.requiredRespLen = compiled_pattern.groups
        self.cmd = cmd
        self.rxRegex = rxRegex
    def SetExpectedResp(self, expected : List):
        if len(expected) == self.requiredRespLen:
            self.expectedResponse = expected
        else:
            raise ValueError(f"Provided response len {len(expected)} does not match expected {self.requiredRespLen}")
    def CheckReceived(self, received : str):
        """
        Return
            None if rx string does not relate to the cmd (rx regex not matched).
            True if rx string matches the regex and matches the expected response (test passes).
            False if rx string matches the regex but does not match the expected response (test fails).
        """
        matchList = list(re.search(self.rxRegex, received).groups())
        if matchList is not None and len(matchList) == self.requiredRespLen:
            if matchList == self.expectedResponse:
                return True
            else:
                return False
        else:
            return None

@dataclass
class SeqTest:
    cmd = None # str or RxCmd
    isRx : bool
    onFail = None
    onPass = None
    def __init__(self, cmd, expResp, /, onFail, onPass):
        self.cmd = cmd
        if isinstance(cmd, RxCmd):
            self.cmd.SetExpectedResp(expResp)
            self.isRx = True
        elif isinstance(cmd, str):
            self.isRx = False
        else:
            raise ValueError("SeqTest cmd must be RxCmd or str")

getAmpState = RxCmd(
    cmd = "get ampstate", # TODO: change for lambda that takes a nb arg and outputs the str + 1
    rxRegex = r"PS(\d+) amp: (\w+)",
)
getNacCodes = RxCmd(
    cmd = "get naccod",
    rxRegex = r"(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)",
)
getCardType = RxCmd(
    cmd = "cardinfo",
    rxRegex = r"Card Type: (\w+)",
)


# Now your sequence lookup works perfectly:
sequence = [
    SeqTest(getAmpState, ["1", "active"], onFail=sys.exit),
    SeqTest("set amp"),
    SeqTest(getAmpState, ["2", "active"])
]

# To delete
test_str = [
    "PS1 amp: active",
    "PS2 amp: active"
]

def main():
    global sequence
    for test in sequence:

        # Send cmd
        ###### serial.write

        # RX path:
        if test.isRx:
            for rx in test_str: # While loop while waiting on serial input
                respCheck = test.cmd.CheckReceived(rx)
                if respCheck is not None:
                    if respCheck:
                        print("PASS")
                        if test.onPass is not None:
                            test.onPass()
                    else:
                        print("FAIL")
                        if test.onFail is not None:
                            test.onFail()


if __name__ == "__main__":
    main()


