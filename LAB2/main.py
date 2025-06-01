import mysql.connector
from tabulate import tabulate
from db_config import db_config

TARIFF_DAY = 2.5
TARIFF_NIGHT = 1.2
DEFAULT_INCREMENT_DAY = 100
DEFAULT_INCREMENT_NIGHT = 80

def connect_db():
    return mysql.connector.connect(**db_config)

def load_all_data(meter_id):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM meters WHERE meter_id = %s ORDER BY date DESC", (meter_id,))
    data = cursor.fetchall()
    conn.close()
    return data

def load_last_data(meter_id):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM meters WHERE meter_id = %s ORDER BY date DESC LIMIT 1", (meter_id,))
    data = cursor.fetchone()
    conn.close()
    return data

def save_data(meter_id, user_day, user_night, adjusted_day, adjusted_night, adjustment_applied, added_kwh_day,
              added_kwh_night, bill):
    conn = connect_db()
    cursor = conn.cursor()
    #print(f"Збереження даних: meter_id={meter_id}, user_day={user_day}, user_night={user_night}, adjusted_day={adjusted_day}, adjusted_night={adjusted_night}, adjustment_applied={adjustment_applied}, added_kwh_day={added_kwh_day}, added_kwh_night={added_kwh_night}, bill={bill}")  # Додано логування
    cursor.execute("""
        INSERT INTO meters (meter_id, user_day, user_night, adjusted_day, adjusted_night, adjustment_applied, added_kwh_day, added_kwh_night, bill) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                   (meter_id, user_day, user_night, adjusted_day, adjusted_night, adjustment_applied, added_kwh_day,
                    added_kwh_night, bill))
    conn.commit()
    conn.close()
    #print("Дані збережено успішно!")  # Додано логування

def process_meter(meter_id, user_day, user_night):
    # Завантажуємо останні дані; якщо немає – беремо значення 0.
    last_data = load_last_data(meter_id) or {"adjusted_day": 0, "adjusted_night": 0}
    old_day, old_night = last_data["adjusted_day"], last_data["adjusted_night"]

    # Обробка денного значення
    if user_day < old_day:
        print("\nВведене денне значення менше попереднього!")
        user_choice = input("Введіть 'Y' для підтвердження використання мінімуму або 'N' для скасування: ")
        if user_choice.upper() == 'Y':
            # Використовуємо попереднє значення для user_day і мінімальний приріст
            user_day = old_day
            adjusted_day = old_day + DEFAULT_INCREMENT_DAY
            added_kwh_day = DEFAULT_INCREMENT_DAY
            adjustment_applied_day = True
        else:
            print("Операцію скасовано.")
            return None
    else:
        adjusted_day = user_day
        added_kwh_day = user_day - old_day
        adjustment_applied_day = False

    # Обробка нічного значення
    if user_night < old_night:
        print("\nВведене нічне значення менше попереднього!")
        user_choice = input("Введіть 'Y' для підтвердження використання мінімуму або 'N' для скасування: ")
        if user_choice.upper() == 'Y':
            user_night = old_night
            adjusted_night = old_night + DEFAULT_INCREMENT_NIGHT
            added_kwh_night = DEFAULT_INCREMENT_NIGHT
            adjustment_applied_night = True
        else:
            print("Операцію скасовано.")
            return None
    else:
        adjusted_night = user_night
        added_kwh_night = user_night - old_night
        adjustment_applied_night = False

    adjustment_applied = adjustment_applied_day or adjustment_applied_night
    day_diff = adjusted_day - old_day
    night_diff = adjusted_night - old_night
    bill = day_diff * TARIFF_DAY + night_diff * TARIFF_NIGHT

    save_data(meter_id, user_day, user_night, adjusted_day, adjusted_night,
              adjustment_applied, added_kwh_day, added_kwh_night, bill)
    return bill

def display_meter_data(meter_id):
    data = load_all_data(meter_id)
    if not data:
        print("Лічильник не знайдено!")
        return

    headers = ["Дата", "Денне споживання", "Нічне споживання", "Скориговане день", "Скориговане ніч", "Коригування",
               "Додано день", "Додано ніч", "Рахунок"]
    table_data = [[row["date"], row["user_day"], row["user_night"], row["adjusted_day"], row["adjusted_night"],
                   "Так" if row["adjustment_applied"] else "Ні", row["added_kwh_day"], row["added_kwh_night"],
                   f"{row['bill']:.2f} грн"] for row in data]

    print("\nДані по лічильнику:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    while True:
        print("\nМеню:")
        print("1. Перевірити дані лічильника")
        print("2. Ввести нові показники")
        print("3. Вихід")
        choice = input("Оберіть дію (1-3): ")

        if choice == "1":
            meter_id = input("Введіть номер лічильника: ")
            display_meter_data(meter_id)
        elif choice == "2":
            meter_id = input("Введіть номер лічильника: ")
            try:
                user_day = int(input("Введіть денне значення: "))
                user_night = int(input("Введіть нічне значення: "))
            except ValueError:
                print("Невірний формат даних! Операцію скасовано.")
                continue

            bill = process_meter(meter_id, user_day, user_night)
            if bill is None:
                print("Повернення до меню.")
                continue
            print(f"Дані збережено! Ваш рахунок: {bill:.2f} грн")
        elif choice == "3":
            print("До побачення!")
            break
        else:
            print("Невірний вибір! Спробуйте ще раз.")

if __name__ == "__main__":
    main()