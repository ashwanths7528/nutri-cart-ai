import sqlite3

DB_PATH = "smart_trolley.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Generate new ID safely
cursor.execute("SELECT product_id FROM products")
ids = [row[0] for row in cursor.fetchall()]

numbers = [
    int(i[1:])
    for i in ids
    if i is not None and isinstance(i, str) and i.startswith("P")
]

if not numbers:
    new_id = "P001"
else:
    new_num = max(numbers) + 1
    new_id = f"P{new_num:03d}"

# Insert proper product
cursor.execute("""
INSERT INTO products (
    product_id,
    product_name,
    category,
    sub_category,
    calories,
    sugar,
    fat,
    fiber,
    protein,
    aisle,
    price
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    new_id,
    "NutriChoice Fiber+",
    "Bakery",
    "biscuits",
    210,   # better calories
    1,     # 🔥 very low sugar
    6,     
    8,     # 🔥 high fiber
    5,
    "Aisle 4",
    40.0
))

conn.commit()

print(f"✅ Clean insert done with ID: {new_id}")

# Verify
cursor.execute("""
SELECT product_id, product_name, sugar, fiber
FROM products
WHERE product_name LIKE '%Nutri%'
""")

rows = cursor.fetchall()

print("\n🔍 Verification:")
for row in rows:
    print(row)

conn.close()