from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import sys

def print_menu():
    print("\n[Q2] 수행하려는 작업에 해당하는 번호를 입력해주세요 (1, 2, 3, 4 중 선택):")
    print("1) collection 조회")
    print("2) collection 생성")
    print("3) collection 이름 변경")
    print("4) collection 삭제")
    print("q) server 종료")

def input_valid_collection(client, prompt):
    while True:
        name = input(prompt).strip()
        if name in client.get_collections().collections:
            return name
        else:
            print("❌ 존재하지 않는 collection입니다. 다시 입력해주세요.")

def main():
    print("[Q1] Qdrant port 입력")
    port_input = input("포트 번호를 입력해주세요 (예: 6333) [기본값: 6333]: ").strip()
    port = 6333 if port_input == "" else int(port_input)

    # Qdrant client 연결
    try:
        client = QdrantClient(host="localhost", port=port)
        client.get_collections()  # 연결 테스트
        print(f"✅ Qdrant 서버({port}번 포트) 연결 성공")
    except Exception as e:
        print(f"❌ Qdrant 서버에 연결할 수 없습니다: {e}")
        sys.exit(1)

    while True:
        print_menu()
        choice = input("입력: ").strip().lower()

        if choice == "1":
            collections = client.get_collections().collections
            if collections:
                print("📦 현재 존재하는 collections:")
                for col in collections:
                    print(f"  - {col.name}")
            else:
                print("📦 현재 존재하는 collection이 없습니다.")

        elif choice == "2":
            name = input("생성하고자 하는 collection 이름을 입력해주세요: ").strip()
            dim_input = input("벡터 차원 수를 입력해주세요 (예: 512) [기본값: 512]: ").strip()
            try:
                dim = 512 if dim_input == "" else int(dim_input)
            except ValueError:
                print("❌ 숫자 형식이 아닙니다.")
                continue

            distance_options = {
                "1": Distance.COSINE,
                "2": Distance.EUCLID,
                "3": Distance.DOT,
                "4": Distance.MANHATTAN
            }
            print("유사도 방식 선택:")
            print("  1) Cosine")
            print("  2) Euclid")
            print("  3) Dot")
            print("  4) Manhattan")
            distance_choice = input("선택 (1~4, 기본값 1): ").strip()
            if distance_choice == "":
                distance_value = Distance.COSINE
            elif distance_choice in distance_options:
                distance_value = distance_options[distance_choice]
            else:
                print("❌ 잘못된 선택입니다. 기본값(Cosine)을 사용합니다.")
                distance_value = Distance.COSINE

            try:
                client.recreate_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=distance_value)
                )
                print(f"✅ '{name}' collection이 생성되었습니다.")
            except Exception as e:
                print(f"❌ collection 생성 실패: {e}")

        elif choice == "3":
            old_name = input("이름을 변경할 collection 이름을 입력해주세요: ").strip()
            collections = [c.name for c in client.get_collections().collections]
            if old_name not in collections:
                print("❌ 존재하지 않는 collection입니다.")
                continue

            new_name = input("새로운 collection 이름을 입력해주세요: ").strip()
            try:
                client.rename_collection(old_collection_name=old_name, new_collection_name=new_name)
                print(f"✅ '{old_name}' → '{new_name}' 로 이름 변경 완료")
            except Exception as e:
                print(f"❌ 이름 변경 실패: {e}")

        elif choice == "4":
            name = input("삭제하려는 collection 이름을 입력해주세요 (또는 'all' 입력 시 전체 삭제): ").strip()
            collections = [c.name for c in client.get_collections().collections]

            if name.lower() == "all":
                confirm = input("⚠️ 모든 collection을 삭제하시겠습니까? (y/n): ").strip().lower()
                if confirm == "y":
                    for col in collections:
                        client.delete_collection(col)
                    print("✅ 모든 collection이 삭제되었습니다.")
                else:
                    print("❌ 삭제 취소")
                continue

            if name not in collections:
                print("❌ 존재하지 않는 collection입니다.")
                continue

            try:
                client.delete_collection(name)
                print(f"✅ '{name}' collection이 삭제되었습니다.")
            except Exception as e:
                print(f"❌ 삭제 실패: {e}")

        elif choice == "q":
            print("🛑 서버 종료")
            break

        else:
            print("❌ 올바르지 않은 입력입니다. 다시 선택해주세요.")

if __name__ == "__main__":
    main()