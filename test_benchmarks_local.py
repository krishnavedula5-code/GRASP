from numerical_lab.benchmarks.loader import load_benchmarks
from numerical_lab.benchmarks.registry import list_ids, list_all


def main() -> None:
    load_benchmarks()

    print("Registered benchmark IDs:")
    print(list_ids())
    print()

    print("Detailed benchmark list:")
    for p in list_all():
        print(
            f"id={p.problem_id}, "
            f"name={p.name}, "
            f"category={p.category}, "
            f"domain={p.domain}"
        )


if __name__ == "__main__":
    main()