def test_assert(condition: bool, message: str = None):
    if not condition:
        # place debug breakpoint here
        print("AssertionError on Test: ", message)
