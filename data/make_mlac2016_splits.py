import os
from datetime import datetime

# -----------------------------
# MLAC2016 speaker split (fixed)
# -----------------------------
TRAIN_SPEAKERS = [
    1, 2, 3, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25,
    27, 33, 35, 36, 37, 38, 39, 45, 46, 47, 48, 50, 53
]

VAL_SPEAKERS = [
    4, 5, 7, 14, 16, 17, 28, 31, 32, 40, 41, 42
]

TEST_SPEAKERS = [
    6, 8, 9, 15, 26, 30, 34, 43, 44, 49, 51, 52
]

# OuluVS2 で欠番扱いするならここ（あなたのベースに合わせて 29 を除外）
EXCLUDE_SPEAKERS = {29}


def list_subdirectories(basedir, speakers, view):
    """
    basedir: lip ディレクトリへのパス（例: ./../lip）
    speakers: [1,2,3,...]
    view: 5 など
    戻り値: 'lip/' からの相対パス（例: '1/5/s1_v5_u31'）のリスト
    """
    subdirectories = []
    missing = []

    for spk in speakers:
        spkpath = os.path.join(basedir, str(spk), str(view))
        if not os.path.isdir(spkpath):
            missing.append(spkpath)
            continue

        for item in os.listdir(spkpath):
            item_path = os.path.join(spkpath, item)
            if os.path.isdir(item_path):
                # lipまでの部分を削除して相対パスを取得
                relative_path = os.path.relpath(item_path, start=basedir)
                subdirectories.append(relative_path)

    if missing:
        print("[WARN] Missing speaker/view directories:")
        for m in missing[:10]:
            print("  ", m)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    return subdirectories


def write_list(path, items):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        for x in items:
            f.write(x + "\n")


def main():
    # ★あなたのベースと同じ指定（data から実行しても動くように相対を維持）
    base_directory = "./../lip"
    profile_view = 5

    # 除外話者を確実に外す
    train_speakers = [s for s in TRAIN_SPEAKERS if s not in EXCLUDE_SPEAKERS]
    val_speakers   = [s for s in VAL_SPEAKERS   if s not in EXCLUDE_SPEAKERS]
    test_speakers  = [s for s in TEST_SPEAKERS  if s not in EXCLUDE_SPEAKERS]

    # ディレクトリ収集（file_listは profile(view=5) を列挙するのが MultiView と整合）
    train_items = list_subdirectories(base_directory, train_speakers, profile_view)
    val_items   = list_subdirectories(base_directory, val_speakers, profile_view)
    test_items  = list_subdirectories(base_directory, test_speakers, profile_view)

    # 出力先（実行場所が data/ でも分かりやすいように ./ に出す）
    out_train = "./mlac2016_train.txt"
    out_val   = "./mlac2016_val.txt"
    out_test  = "./mlac2016_test.txt"

    write_list(out_train, train_items)
    write_list(out_val, val_items)
    write_list(out_test, test_items)

    # 分割情報ログ
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    info_path = f"./mlac2016_split_info_{ts}.txt"
    with open(info_path, "w") as f:
        f.write("[MLAC2016 speaker split]\n")
        f.write(f"base_directory: {os.path.abspath(base_directory)}\n")
        f.write(f"profile_view  : {profile_view}\n")
        f.write(f"train_speakers({len(train_speakers)}): {train_speakers}\n")
        f.write(f"val_speakers  ({len(val_speakers)}): {val_speakers}\n")
        f.write(f"test_speakers ({len(test_speakers)}): {test_speakers}\n")
        f.write("\n")
        f.write(f"train_items: {len(train_items)} -> {os.path.abspath(out_train)}\n")
        f.write(f"val_items  : {len(val_items)} -> {os.path.abspath(out_val)}\n")
        f.write(f"test_items : {len(test_items)} -> {os.path.abspath(out_test)}\n")

    # 標準出力（「何も出ない」を潰すため、必ず表示）
    print("\n[OUTPUT]")
    print("  train:", len(train_items), "->", os.path.abspath(out_train))
    print("  val  :", len(val_items),   "->", os.path.abspath(out_val))
    print("  test :", len(test_items),  "->", os.path.abspath(out_test))
    print("  info :", os.path.abspath(info_path))

    # 件数0のときは分かるように強めに警告
    if len(train_items) == 0 or len(val_items) == 0 or len(test_items) == 0:
        print("\n[ERROR] One of the splits has 0 items.")
        print("Check that base_directory and profile_view are correct, and lip/<spk>/<view>/ exists.")


if __name__ == "__main__":
    main()
