import asyncio
import aiosqlite
from pathlib import Path

async def inspect_db():
    db_path = Path("data/flow.db")
    if not db_path.exists():
        print("Database not found!")
        return

    print(f"Inspecting database: {db_path}")
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT id, email, user_paygate_tier, credits FROM tokens") as cursor:
            rows = await cursor.fetchall()
            print(f"Found {len(rows)} tokens:")
            for row in rows:
                print(f"ID: {row['id']}")
                print(f"  Email: {row['email']}")
                print(f"  Tier: {row['user_paygate_tier']}")
                print(f"  Credits: {row['credits']}")
                print("-" * 20)

if __name__ == "__main__":
    asyncio.run(inspect_db())
