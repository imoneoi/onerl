import os
import argparse


def convert_profile_log(path: str, buffering: int = 1048576, max_str_len: int = 16):
    # open json file
    json_file = open(os.path.join(path, "profile.json"), "wt", buffering=buffering)
    json_file.write("[")

    # enumerate log files
    for fn in os.listdir(path):
        log_file_name = os.path.join(path, fn)
        if (not os.path.isfile(log_file_name)) or (fn == "profile.json"):
            continue

        # read all
        print("Reading ", fn)
        with open(log_file_name, "rb") as f:
            log = f.read()
            f.close()

        print("Converting ", fn)
        log = memoryview(log)
        idx = 0

        last_state = None
        while True:
            timestamp = int.from_bytes(log[idx: idx + 8], "big")
            idx += 8
            if not timestamp:
                break
            state_bytes = log[idx: idx + max_str_len].tobytes()
            state_end = state_bytes.find(0)
            idx += state_end + 1
            state = state_bytes[:state_end].decode()

            # write log
            timestamp //= 1000
            if last_state is not None:
                json_file.write('{"name":"' + last_state + '","ph":"E","pid":"Main","tid":"' + fn + '","ts":' + str(timestamp) + '},')
            json_file.write('{"name":"' + state + '","ph":"B","pid":"Main","tid":"' + fn + '","ts":' + str(timestamp) + '},')
            last_state = state

    json_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to log to convert")

    args = parser.parse_args()
    convert_profile_log(args.path)


if __name__ == "__main__":
    main()
