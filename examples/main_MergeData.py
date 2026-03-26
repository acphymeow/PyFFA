import argparse
from FFAG_dump import merge_files_in_folder


def main():
    parser = argparse.ArgumentParser(description="Merge Dump Data")

    # 添加输入文件夹参数
    parser.add_argument('input_folder', type=str, help="Input folder containing .csv files to be merged.")

    args = parser.parse_args()

    # 调用合并函数，生成合并后的文件
    merge_files_in_folder(args.input_folder)
    print(f"Merged files successfully into {args.input_folder}.")


if __name__ == '__main__':
    main()
