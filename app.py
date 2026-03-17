import os

print("1. Build knowledge base")
print("2. Ask questions")

choice = input("\nChoose 1 or 2: ").strip()

if choice == "1":
    os.system("python ingest.py")
elif choice == "2":
    os.system("python query.py")
else:
    print("Invalid choice")