import sys
import argparse

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)

    args = parser.parse_args()

    user_input=args.prompt

    prompt = f"""User input: {user_input}

Your response:"""

    print(prompt)


if __name__ == "__main__":
    main()
