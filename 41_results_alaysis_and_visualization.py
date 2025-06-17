# pip install pandas matplotlib seaborn scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from pathlib import Path
import glob
import sys

# -------------------- CSV 목록 선택 --------------------

def select_result_csv():
    result_files = sorted(Path("results").rglob("result_*.csv"))
    if not result_files:
        print("❌ 분석 가능한 CSV 파일이 존재하지 않습니다.")
        sys.exit(1)

    print("\n📁 분석할 CSV 파일을 선택하세요:")
    for idx, path in enumerate(result_files, start=1):
        print(f"[{idx}] {path}")

    while True:
        try:
            choice = int(input("\n선택한 파일 번호를 입력하세요: "))
            if 1 <= choice <= len(result_files):
                return result_files[choice - 1]
            else:
                print("⚠️ 유효한 번호를 입력하세요.")
        except ValueError:
            print("⚠️ 숫자만 입력하세요.")

# -------------------- 데이터 로드 --------------------

result_data_path = select_result_csv()
print(f"\n📂 선택된 결과 CSV: {result_data_path}")
results_df = pd.read_csv(result_data_path)
print("✅ 데이터 로드 완료. 총 샘플 수:", len(results_df))

# 저장 경로 설정
output_dir = result_data_path.parent
output_img_dir = output_dir
output_csv_dir = output_dir

# -------------------- 분석 및 시각화 --------------------

groups = results_df.groupby(['case', 'delegate_type'])
class_list = sorted(results_df['true_class'].unique())

# [1] Confusion Matrix
print("\n[1] Confusion Matrix 생성 및 시각화")
for (case, dtype), group_df in groups:
    y_true = group_df['true_class']
    y_pred = group_df['predicted_class']
    cm = confusion_matrix(y_true, y_pred, labels=class_list)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_list)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f'Confusion Matrix\n{case.upper()} - {dtype}')
    plt.tight_layout()

    out_path = output_img_dir / f"cm_{case}_{dtype}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  - 저장 완료: {out_path}")

# [2] 유사도 점수 분포 시각화
print("\n[2] 유사도 점수 분포 시각화")
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.boxplot(data=results_df, x="delegate_type", y="similarity_score", hue="case")
plt.title("대표 벡터 유형별 유사도 점수 분포")
plt.ylabel("Cosine Similarity")
plt.xlabel("Delegate Vector Type")
plt.legend(title="Case")
plt.tight_layout()
score_dist_path = output_img_dir / "similarity_score_distribution.png"
plt.savefig(score_dist_path)
plt.close()
print(f"  - 저장 완료: {score_dist_path}")

# [3] 분류 성능 리포트
print("\n[3] 분류 성능 리포트 (Precision / Recall / F1 Score)")
for (case, dtype), group_df in groups:
    print(f"\n📘 [{case.upper()} - {dtype}] 결과:")
    report_dict = classification_report(
        group_df['true_class'], group_df['predicted_class'], labels=class_list, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.insert(0, 'case', case)
    report_df.insert(1, 'delegate_type', dtype)

    csv_path = output_csv_dir / f"metrics_{case}_{dtype}.csv"
    report_df.to_csv(csv_path, float_format='%.4f')
    print(report_df.round(4))
    print(f"  - 저장 완료: {csv_path}")

print("\n✅ 분석 및 시각화 완료.")