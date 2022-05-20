import os
import argparse


def convert_profile_log(path: str, buffering: int = 1048576, max_str_len: int = 16):
    # open json file
    json_file = open(os.path.join(path, "profile.json"), "wt", buffering=buffering)
    json_file.write("[")

    # pid & tid map
    pid_map = {}
    pid_no = 1
    tid_map = {}
    tid_no = 10000

    # enumerate log files
    log_file_list = sorted(os.listdir(path))
    for fn in log_file_list:
        log_file_name = os.path.join(path, fn)
        if (not os.path.isfile(log_file_name)) or (fn == "profile.json"):
            continue

        # read all
        print("Reading ", fn)
        with open(log_file_name, "rb") as f:
            log = f.read()
            f.close()

        print("Converting ", fn)
        # assign pid & tid
        node_ns, node_name = fn.split("@")

        if node_ns not in pid_map:
            pid_map[node_ns] = pid_no
            json_file.write('{"pid":' + str(pid_no) + ',"tid":0,"ts":0,"ph":"M","cat":"__metadata","name":"process_name","args":{"name":"' + node_ns + '"}},')

            pid_no += 1
        pid = str(pid_map[node_ns])

        if fn not in tid_map:
            tid_map[fn] = tid_no
            json_file.write('{"pid":' + pid + ',"tid":' + str(tid_no) + ',"ts":0,"ph":"M","cat":"__metadata","name":"thread_name","args":{"name":"' + node_name + '"}},')

            tid_no += 1
        tid = str(tid_map[fn])
            
        # convert
        log_len = len(log)
        log = memoryview(log)
        idx = 0

        last_state = None
        while True:
            timestamp = int.from_bytes(log[idx: idx + 8], "big")
            idx += 8
            if idx > log_len:
                break

            state_bytes = log[idx: idx + max_str_len].tobytes()
            state_end = state_bytes.find(0)
            idx += state_end + 1
            if idx > log_len:
                break
            state = state_bytes[:state_end].decode()

            # write log
            timestamp //= 1000
            if last_state is not None:
                json_file.write('{"name":"' + last_state + '","ph":"E","pid":' + pid + ',"tid":' + tid + ',"ts":' + str(timestamp) + '},')
            json_file.write('{"name":"' + state + '","ph":"B","pid":' + pid + ',"tid":' + tid + ',"ts":' + str(timestamp) + '},')

            last_state = state
            last_timestamp = timestamp
        # end last event
        if last_state is not None:
            json_file.write('{"name":"' + last_state + '","ph":"E","pid":' + pid + ',"tid":' + tid + ',"ts":' + str(last_timestamp) + '},')

    json_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to log to convert")

    args = parser.parse_args()
    convert_profile_log(args.path)


if __name__ == "__main__":
    main()
