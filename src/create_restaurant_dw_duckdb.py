from __future__ import annotations

import argparse
import random
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import duckdb


DB_PATH = "restaurant_dw.duckdb"


DROP_ORDER = [
    "Fact_Sales",
    "Dim_Customer",
    "Dim_Menu_Item",
    "Dim_Restaurant",
    "Dim_Time",
    "Dim_Date",
]


CREATE_TABLES = [
    """
    CREATE TABLE Dim_Date
    (
        Date_ID        INTEGER      NOT NULL,
        Full_Date      DATE         NOT NULL,
        Day_Number     INTEGER      NOT NULL,
        Month_Number   INTEGER      NOT NULL,
        Month_Name     VARCHAR      NOT NULL,
        Quarter_Number INTEGER      NOT NULL,
        Year_Number    INTEGER      NOT NULL,
        Day_Of_Week    INTEGER      NOT NULL,
        Is_Weekend     BOOLEAN      NOT NULL,
        Is_Holiday     BOOLEAN      NOT NULL,
        Season         VARCHAR      NOT NULL,

        CONSTRAINT PK_Dim_Date PRIMARY KEY (Date_ID),
        CONSTRAINT UQ_Dim_Date_Full_Date UNIQUE (Full_Date),
        CONSTRAINT CHK_Dim_Date_Day_Number CHECK (Day_Number BETWEEN 1 AND 31),
        CONSTRAINT CHK_Dim_Date_Month_Number CHECK (Month_Number BETWEEN 1 AND 12),
        CONSTRAINT CHK_Dim_Date_Quarter_Number CHECK (Quarter_Number BETWEEN 1 AND 4),
        CONSTRAINT CHK_Dim_Date_Day_Of_Week CHECK (Day_Of_Week BETWEEN 1 AND 7),
        CONSTRAINT CHK_Dim_Date_Season CHECK (Season IN ('Зима', 'Весна', 'Лето', 'Осень'))
    )
    """,
    """
    CREATE TABLE Dim_Time
    (
        Time_ID       INTEGER NOT NULL,
        Hour_Value    INTEGER NOT NULL,
        Minute_Value  INTEGER NOT NULL,
        Time_Interval VARCHAR NOT NULL,
        Part_Of_Day   VARCHAR NOT NULL,

        CONSTRAINT PK_Dim_Time PRIMARY KEY (Time_ID),
        CONSTRAINT CHK_Dim_Time_Hour CHECK (Hour_Value BETWEEN 0 AND 23),
        CONSTRAINT CHK_Dim_Time_Minute CHECK (Minute_Value BETWEEN 0 AND 59)
    )
    """,
    """
    CREATE TABLE Dim_Restaurant
    (
        Restaurant_ID        INTEGER NOT NULL,
        Restaurant_Name      VARCHAR NOT NULL,
        City                 VARCHAR NOT NULL,
        District             VARCHAR,
        Address              VARCHAR NOT NULL,
        Restaurant_Type      VARCHAR NOT NULL,
        Opening_Date         DATE,
        Seating_Capacity     INTEGER NOT NULL,
        Delivery_Available   BOOLEAN NOT NULL,
        Drive_Thru_Available BOOLEAN NOT NULL,

        CONSTRAINT PK_Dim_Restaurant PRIMARY KEY (Restaurant_ID),
        CONSTRAINT CHK_Dim_Restaurant_Seating_Capacity CHECK (Seating_Capacity >= 0)
    )
    """,
    """
    CREATE TABLE Dim_Menu_Item
    (
        Menu_Item_ID     INTEGER NOT NULL,
        Restaurant_ID    INTEGER NOT NULL,
        Menu_Item_Name   VARCHAR NOT NULL,
        Category_Name    VARCHAR NOT NULL,
        Subcategory_Name VARCHAR,
        Cuisine_Type     VARCHAR NOT NULL,
        Portion_Size     VARCHAR,
        Calories         INTEGER,
        Base_Cost        DECIMAL(10, 2) NOT NULL,
        Standard_Price   DECIMAL(10, 2) NOT NULL,
        Is_Seasonal      BOOLEAN NOT NULL,
        Launch_Date      DATE,
        Is_Active        BOOLEAN NOT NULL,

        CONSTRAINT PK_Dim_Menu_Item PRIMARY KEY (Menu_Item_ID),
        CONSTRAINT FK_Dim_Menu_Item_Dim_Restaurant FOREIGN KEY (Restaurant_ID) REFERENCES Dim_Restaurant(Restaurant_ID),
        CONSTRAINT CHK_Dim_Menu_Item_Calories CHECK (Calories IS NULL OR Calories >= 0),
        CONSTRAINT CHK_Dim_Menu_Item_Base_Cost CHECK (Base_Cost >= 0),
        CONSTRAINT CHK_Dim_Menu_Item_Standard_Price CHECK (Standard_Price >= 0)
    )
    """,
    """
    CREATE TABLE Dim_Customer
    (
        Customer_ID         INTEGER NOT NULL,
        Gender              VARCHAR,
        Age_Group           VARCHAR,
        Loyalty_Status      VARCHAR,
        Registration_Date   DATE,
        Preferred_Channel   VARCHAR,
        Segment             VARCHAR,
        City                VARCHAR,
        Average_Check_Group VARCHAR,

        CONSTRAINT PK_Dim_Customer PRIMARY KEY (Customer_ID),
        CONSTRAINT CHK_Dim_Customer_Gender CHECK (Gender IS NULL OR Gender IN ('Мужской', 'Женский', 'Иной'))
    )
    """,
    """
    CREATE TABLE Fact_Sales
    (
        Sales_Fact_ID    BIGINT NOT NULL,
        Date_ID          INTEGER NOT NULL,
        Time_ID          INTEGER NOT NULL,
        Restaurant_ID    INTEGER NOT NULL,
        Menu_Item_ID     INTEGER NOT NULL,
        Customer_ID      INTEGER NOT NULL,
        Quantity         INTEGER NOT NULL,
        Unit_Price       DECIMAL(10, 2) NOT NULL,
        Discount_Amount  DECIMAL(10, 2) NOT NULL,
        Cost_Amount      DECIMAL(10, 2) NOT NULL,
        Revenue_Amount   DECIMAL(10, 2) NOT NULL,
        Profit_Amount    DECIMAL(10, 2) NOT NULL,
        Preparation_Time INTEGER,
        Service_Time     INTEGER,
        Rating_Value     DECIMAL(3, 2),

        CONSTRAINT PK_Fact_Sales PRIMARY KEY (Sales_Fact_ID),
        CONSTRAINT FK_Fact_Sales_Dim_Date FOREIGN KEY (Date_ID) REFERENCES Dim_Date(Date_ID),
        CONSTRAINT FK_Fact_Sales_Dim_Time FOREIGN KEY (Time_ID) REFERENCES Dim_Time(Time_ID),
        CONSTRAINT FK_Fact_Sales_Dim_Restaurant FOREIGN KEY (Restaurant_ID) REFERENCES Dim_Restaurant(Restaurant_ID),
        CONSTRAINT FK_Fact_Sales_Dim_Menu_Item FOREIGN KEY (Menu_Item_ID) REFERENCES Dim_Menu_Item(Menu_Item_ID),
        CONSTRAINT FK_Fact_Sales_Dim_Customer FOREIGN KEY (Customer_ID) REFERENCES Dim_Customer(Customer_ID),
        CONSTRAINT CHK_Fact_Sales_Quantity CHECK (Quantity > 0),
        CONSTRAINT CHK_Fact_Sales_Unit_Price CHECK (Unit_Price >= 0),
        CONSTRAINT CHK_Fact_Sales_Discount_Amount CHECK (Discount_Amount >= 0),
        CONSTRAINT CHK_Fact_Sales_Cost_Amount CHECK (Cost_Amount >= 0),
        CONSTRAINT CHK_Fact_Sales_Revenue_Amount CHECK (Revenue_Amount >= 0),
        CONSTRAINT CHK_Fact_Sales_Preparation_Time CHECK (Preparation_Time IS NULL OR Preparation_Time >= 0),
        CONSTRAINT CHK_Fact_Sales_Service_Time CHECK (Service_Time IS NULL OR Service_Time >= 0),
        CONSTRAINT CHK_Fact_Sales_Rating_Value CHECK (Rating_Value IS NULL OR Rating_Value BETWEEN 0 AND 5)
    )
    """,
]


CREATE_INDEXES = [
    "CREATE INDEX IX_Fact_Sales_Date_ID ON Fact_Sales(Date_ID)",
    "CREATE INDEX IX_Fact_Sales_Time_ID ON Fact_Sales(Time_ID)",
    "CREATE INDEX IX_Fact_Sales_Restaurant_ID ON Fact_Sales(Restaurant_ID)",
    "CREATE INDEX IX_Fact_Sales_Menu_Item_ID ON Fact_Sales(Menu_Item_ID)",
    "CREATE INDEX IX_Fact_Sales_Customer_ID ON Fact_Sales(Customer_ID)",
    "CREATE INDEX IX_Fact_Sales_Date_Restaurant_Menu ON Fact_Sales(Date_ID, Restaurant_ID, Menu_Item_ID)",
]


CREATE_VIEWS = [
    """
    CREATE VIEW vw_daily_sales AS
    SELECT
        d.Full_Date,
        d.Year_Number,
        d.Month_Number,
        d.Day_Number,
        r.Restaurant_Name,
        r.City,
        SUM(f.Quantity) AS Total_Quantity,
        SUM(f.Revenue_Amount) AS Total_Revenue,
        SUM(f.Profit_Amount) AS Total_Profit,
        AVG(f.Rating_Value) AS Average_Rating
    FROM Fact_Sales f
    JOIN Dim_Date d ON f.Date_ID = d.Date_ID
    JOIN Dim_Restaurant r ON f.Restaurant_ID = r.Restaurant_ID
    GROUP BY
        d.Full_Date,
        d.Year_Number,
        d.Month_Number,
        d.Day_Number,
        r.Restaurant_Name,
        r.City
    """,
    """
    CREATE VIEW vw_menu_sales AS
    SELECT
        r.Restaurant_Name,
        r.City,
        m.Category_Name,
        m.Subcategory_Name,
        m.Menu_Item_Name,
        SUM(f.Quantity) AS Total_Quantity,
        SUM(f.Revenue_Amount) AS Total_Revenue,
        SUM(f.Profit_Amount) AS Total_Profit,
        AVG(f.Rating_Value) AS Average_Rating
    FROM Fact_Sales f
    JOIN Dim_Menu_Item m ON f.Menu_Item_ID = m.Menu_Item_ID
    JOIN Dim_Restaurant r ON f.Restaurant_ID = r.Restaurant_ID
    GROUP BY
        r.Restaurant_Name,
        r.City,
        m.Category_Name,
        m.Subcategory_Name,
        m.Menu_Item_Name
    """,
    """
    CREATE VIEW vw_restaurant_menu AS
    SELECT
        r.Restaurant_ID,
        r.Restaurant_Name,
        r.City,
        COUNT(*) AS Menu_Items_Count,
        SUM(CASE WHEN m.Is_Seasonal THEN 1 ELSE 0 END) AS Seasonal_Items_Count
    FROM Dim_Restaurant r
    JOIN Dim_Menu_Item m ON r.Restaurant_ID = m.Restaurant_ID
    GROUP BY
        r.Restaurant_ID,
        r.Restaurant_Name,
        r.City
    """,
    """
    CREATE VIEW vw_customer_flow AS
    SELECT
        d.Full_Date,
        d.Year_Number,
        d.Month_Number,
        d.Season,
        d.Is_Weekend,
        r.Restaurant_Name,
        r.City,
        COUNT(*) AS Order_Line_Count,
        COUNT(DISTINCT f.Customer_ID) AS Unique_Customers,
        SUM(f.Quantity) AS Total_Items_Sold,
        SUM(f.Revenue_Amount) AS Total_Revenue,
        SUM(f.Profit_Amount) AS Total_Profit
    FROM Fact_Sales f
    JOIN Dim_Date d ON f.Date_ID = d.Date_ID
    JOIN Dim_Restaurant r ON f.Restaurant_ID = r.Restaurant_ID
    GROUP BY
        d.Full_Date,
        d.Year_Number,
        d.Month_Number,
        d.Season,
        d.Is_Weekend,
        r.Restaurant_Name,
        r.City
    """,
]


def recreate_schema(conn: duckdb.DuckDBPyConnection) -> None:
    for view_name in ["vw_customer_flow", "vw_restaurant_menu", "vw_menu_sales", "vw_daily_sales"]:
        conn.execute(f"DROP VIEW IF EXISTS {view_name}")

    for table_name in DROP_ORDER:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

    for statement in CREATE_TABLES:
        conn.execute(statement)

    for statement in CREATE_INDEXES:
        conn.execute(statement)

    for statement in CREATE_VIEWS:
        conn.execute(statement)


def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Зима"
    if month in (3, 4, 5):
        return "Весна"
    if month in (6, 7, 8):
        return "Лето"
    return "Осень"


def get_part_of_day(hour: int) -> str:
    if 6 <= hour <= 11:
        return "Утро"
    if 12 <= hour <= 16:
        return "День"
    if 17 <= hour <= 22:
        return "Вечер"
    return "Ночь"


def get_time_interval(hour: int, minute: int) -> str:
    start_minute = 0 if minute < 30 else 30
    end_hour = hour if start_minute == 0 else (hour + 1) % 24
    end_minute = 30 if start_minute == 0 else 0
    return f"{hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}"


def date_id(value: date) -> int:
    return int(value.strftime("%Y%m%d"))


def time_id(value: time) -> int:
    return value.hour * 100 + value.minute


def fill_date_dimension(
    conn: duckdb.DuckDBPyConnection,
    start_date: str = "2024-01-01",
    end_date: str = "2026-12-31",
) -> None:
    month_names = {
        1: "Январь",
        2: "Февраль",
        3: "Март",
        4: "Апрель",
        5: "Май",
        6: "Июнь",
        7: "Июль",
        8: "Август",
        9: "Сентябрь",
        10: "Октябрь",
        11: "Ноябрь",
        12: "Декабрь",
    }

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    current = start
    rows = []

    while current <= end:
        rows.append(
            (
                date_id(current),
                current,
                current.day,
                current.month,
                month_names[current.month],
                (current.month - 1) // 3 + 1,
                current.year,
                current.isoweekday(),
                current.isoweekday() in (6, 7),
                False,
                get_season(current.month),
            )
        )
        current += timedelta(days=1)

    conn.executemany(
        """
        INSERT INTO Dim_Date
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def fill_time_dimension(conn: duckdb.DuckDBPyConnection) -> None:
    rows = []

    for hour in range(24):
        for minute in range(60):
            value = time(hour, minute)
            rows.append(
                (
                    time_id(value),
                    hour,
                    minute,
                    get_time_interval(hour, minute),
                    get_part_of_day(hour),
                )
            )

    conn.executemany(
        """
        INSERT INTO Dim_Time
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )


def fill_demo_data(conn: duckdb.DuckDBPyConnection) -> None:
    rng = random.Random(42)

    def money(value: Decimal) -> Decimal:
        return value.quantize(Decimal("0.01"))

    def random_date(start: date, end: date) -> date:
        days = (end - start).days
        return start + timedelta(days=rng.randint(0, days))

    def choose_visit_time() -> time:
        periods = [
            (8, 10, 0.08),
            (11, 15, 0.36),
            (16, 17, 0.10),
            (18, 22, 0.46),
        ]
        start_hour, end_hour, _ = rng.choices(periods, weights=[item[2] for item in periods], k=1)[0]
        return time(rng.randint(start_hour, end_hour), rng.randint(0, 59))

    restaurants = [
        (1, "Ресторан Центр", "Москва", "Центральный", "ул. Тверская, 1", "Casual Dining", date(2021, 5, 20), 90, True, False),
        (2, "Ресторан Север", "Москва", "Северный", "ул. Лесная, 15", "Fast Casual", date(2022, 3, 10), 65, True, True),
        (3, "Ресторан Юг", "Москва", "Южный", "Каширское шоссе, 24", "Family Restaurant", date(2023, 9, 1), 80, True, False),
    ]

    menu_catalog = {
        1: [
            ("Паста Карбонара", "Основные блюда", "Паста", "Итальянская", "350 г", 720, "190.00", "520.00", False, None),
            ("Паста Болоньезе", "Основные блюда", "Паста", "Итальянская", "360 г", 760, "210.00", "560.00", False, None),
            ("Лазанья", "Основные блюда", "Запеканки", "Итальянская", "330 г", 810, "230.00", "610.00", False, None),
            ("Ризотто с грибами", "Основные блюда", "Ризотто", "Итальянская", "320 г", 640, "200.00", "570.00", False, None),
            ("Пицца Маргарита", "Основные блюда", "Пицца", "Итальянская", "410 г", 850, "240.00", "650.00", False, None),
            ("Цезарь с курицей", "Салаты", "Салаты с птицей", "Европейская", "280 г", 430, "150.00", "420.00", False, None),
            ("Минестроне", "Супы", "Овощные супы", "Итальянская", "300 г", 270, "95.00", "310.00", False, None),
            ("Тирамису", "Десерты", "Холодные десерты", "Итальянская", "160 г", 360, "110.00", "330.00", False, None),
            ("Панна-котта", "Десерты", "Холодные десерты", "Итальянская", "150 г", 320, "95.00", "300.00", False, None),
            ("Домашний лимонад", "Напитки", "Холодные напитки", "Европейская", "400 мл", 120, "45.00", "190.00", False, None),
            ("Летний гаспачо", "Супы", "Холодные супы", "Испанская", "280 г", 210, "90.00", "290.00", True, "Лето"),
            ("Тыквенный крем-суп", "Супы", "Крем-супы", "Европейская", "300 г", 310, "105.00", "330.00", True, "Осень"),
        ],
        2: [
            ("Борщ с говядиной", "Супы", "Мясные супы", "Русская", "350 г", 390, "130.00", "380.00", False, None),
            ("Солянка", "Супы", "Мясные супы", "Русская", "330 г", 430, "145.00", "410.00", False, None),
            ("Пельмени домашние", "Основные блюда", "Пельмени", "Русская", "320 г", 680, "180.00", "480.00", False, None),
            ("Котлета по-киевски", "Основные блюда", "Блюда из птицы", "Европейская", "310 г", 740, "220.00", "590.00", False, None),
            ("Стейк из лосося", "Основные блюда", "Рыба", "Скандинавская", "280 г", 610, "360.00", "890.00", False, None),
            ("Салат Оливье", "Салаты", "Мясные салаты", "Русская", "250 г", 410, "120.00", "350.00", False, None),
            ("Сельдь под шубой", "Салаты", "Рыбные салаты", "Русская", "260 г", 460, "125.00", "360.00", False, None),
            ("Сырники со сметаной", "Завтраки", "Творожные блюда", "Русская", "220 г", 520, "95.00", "310.00", False, None),
            ("Медовик", "Десерты", "Торты", "Русская", "150 г", 430, "85.00", "280.00", False, None),
            ("Клюквенный морс", "Напитки", "Холодные напитки", "Русская", "400 мл", 130, "40.00", "170.00", False, None),
            ("Окрошка летняя", "Супы", "Холодные супы", "Русская", "330 г", 300, "90.00", "290.00", True, "Лето"),
            ("Глинтвейн безалкогольный", "Напитки", "Горячие напитки", "Европейская", "300 мл", 160, "55.00", "220.00", True, "Зима"),
        ],
        3: [
            ("Хачапури по-аджарски", "Основные блюда", "Выпечка", "Кавказская", "360 г", 820, "210.00", "560.00", False, None),
            ("Хинкали", "Основные блюда", "Тесто и мясо", "Кавказская", "5 шт", 700, "190.00", "520.00", False, None),
            ("Шашлык из курицы", "Основные блюда", "Шашлык", "Кавказская", "300 г", 620, "230.00", "610.00", False, None),
            ("Люля-кебаб", "Основные блюда", "Блюда на гриле", "Кавказская", "280 г", 690, "220.00", "590.00", False, None),
            ("Чахохбили", "Основные блюда", "Рагу", "Грузинская", "330 г", 580, "185.00", "510.00", False, None),
            ("Лобио", "Основные блюда", "Бобовые", "Грузинская", "300 г", 430, "110.00", "340.00", False, None),
            ("Харчо", "Супы", "Мясные супы", "Грузинская", "350 г", 410, "140.00", "390.00", False, None),
            ("Салат с сулугуни", "Салаты", "Сырные салаты", "Кавказская", "260 г", 390, "135.00", "370.00", False, None),
            ("Пахлава", "Десерты", "Восточные сладости", "Восточная", "120 г", 450, "75.00", "260.00", False, None),
            ("Айран", "Напитки", "Кисломолочные напитки", "Кавказская", "300 мл", 95, "35.00", "150.00", False, None),
            ("Арбузный салат", "Салаты", "Сезонные салаты", "Европейская", "250 г", 210, "80.00", "280.00", True, "Лето"),
            ("Согревающий чай", "Напитки", "Горячие напитки", "Восточная", "400 мл", 140, "50.00", "210.00", True, "Зима"),
        ],
    }

    menu_items = []
    restaurant_menu_items = {restaurant_id: [] for restaurant_id in menu_catalog}
    menu_item_id = 1

    for restaurant_id, items in menu_catalog.items():
        for item in items:
            (
                name,
                category,
                subcategory,
                cuisine,
                portion,
                calories,
                base_cost,
                standard_price,
                is_seasonal,
                item_season,
            ) = item
            launch_date = date(2021, 6, 1) + timedelta(days=menu_item_id * 12)
            row = (
                menu_item_id,
                restaurant_id,
                name,
                category,
                subcategory,
                cuisine,
                portion,
                calories,
                Decimal(base_cost),
                Decimal(standard_price),
                is_seasonal,
                launch_date,
                True,
            )
            menu_items.append(row)
            restaurant_menu_items[restaurant_id].append(
                {
                    "id": menu_item_id,
                    "base_cost": Decimal(base_cost),
                    "standard_price": Decimal(standard_price),
                    "is_seasonal": is_seasonal,
                    "season": item_season,
                }
            )
            menu_item_id += 1

    genders = ["Мужской", "Женский", "Иной"]
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    loyalty_statuses = ["None", "Bronze", "Silver", "Gold"]
    preferred_channels = ["Зал", "Доставка", "Самовывоз"]
    segments = ["Новый клиент", "Постоянный клиент", "Акционный клиент", "Корпоративный клиент"]
    average_check_groups = ["Низкий", "Средний", "Высокий"]

    customers = []
    customer_profiles = {}

    for customer_id in range(1, 501):
        loyalty_status = rng.choices(loyalty_statuses, weights=[0.35, 0.30, 0.22, 0.13], k=1)[0]
        average_check_group = rng.choices(average_check_groups, weights=[0.30, 0.50, 0.20], k=1)[0]
        segment = rng.choices(segments, weights=[0.28, 0.42, 0.22, 0.08], k=1)[0]
        row = (
            customer_id,
            rng.choices(genders, weights=[0.48, 0.50, 0.02], k=1)[0],
            rng.choices(age_groups, weights=[0.20, 0.34, 0.24, 0.14, 0.08], k=1)[0],
            loyalty_status,
            random_date(date(2021, 1, 1), date(2025, 12, 31)),
            rng.choices(preferred_channels, weights=[0.58, 0.30, 0.12], k=1)[0],
            segment,
            "Москва",
            average_check_group,
        )
        customers.append(row)
        customer_profiles[customer_id] = {
            "loyalty_status": loyalty_status,
            "average_check_group": average_check_group,
            "segment": segment,
        }

    conn.executemany("INSERT INTO Dim_Restaurant VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", restaurants)
    conn.executemany("INSERT INTO Dim_Menu_Item VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", menu_items)
    conn.executemany("INSERT INTO Dim_Customer VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", customers)

    dimension_start, dimension_end = conn.execute("SELECT MIN(Full_Date), MAX(Full_Date) FROM Dim_Date").fetchone()
    sales_start = max(dimension_start, date(2025, 1, 1))
    sales_end = min(dimension_end, date(2025, 12, 31))

    if sales_start > sales_end:
        sales_start = dimension_start
        sales_end = min(dimension_end, dimension_start + timedelta(days=364))

    customer_ids = list(customer_profiles)
    customer_weights = []

    for customer_id in customer_ids:
        profile = customer_profiles[customer_id]
        loyalty_weight = {
            "Gold": 3.20,
            "Silver": 2.20,
            "Bronze": 1.35,
            "None": 1.00,
        }[profile["loyalty_status"]]
        customer_weights.append(loyalty_weight)

    season_multipliers = {
        "Зима": Decimal("0.82"),
        "Весна": Decimal("1.00"),
        "Лето": Decimal("1.30"),
        "Осень": Decimal("1.12"),
    }

    restaurant_base_flow = {
        1: Decimal("58"),
        2: Decimal("45"),
        3: Decimal("52"),
    }

    restaurant_season_adjustment = {
        1: {"Зима": Decimal("0.96"), "Весна": Decimal("1.00"), "Лето": Decimal("1.06"), "Осень": Decimal("1.03")},
        2: {"Зима": Decimal("1.12"), "Весна": Decimal("1.00"), "Лето": Decimal("0.95"), "Осень": Decimal("1.04")},
        3: {"Зима": Decimal("0.88"), "Весна": Decimal("1.04"), "Лето": Decimal("1.20"), "Осень": Decimal("1.02")},
    }

    loyalty_discount = {
        "Gold": Decimal("0.07"),
        "Silver": Decimal("0.04"),
        "Bronze": Decimal("0.02"),
        "None": Decimal("0.00"),
    }

    rating_base = {
        1: 4.55,
        2: 4.35,
        3: 4.45,
    }

    sales = []
    sales_fact_id = 1
    current = sales_start

    while current <= sales_end:
        season = get_season(current.month)
        weekday = current.isoweekday()
        weekend_multiplier = Decimal("1.25") if weekday in (6, 7) else Decimal("1.00")
        friday_multiplier = Decimal("1.12") if weekday == 5 else Decimal("1.00")
        monday_multiplier = Decimal("0.90") if weekday == 1 else Decimal("1.00")

        for restaurant_id in restaurant_base_flow:
            random_multiplier = Decimal(str(round(rng.uniform(0.84, 1.18), 2)))
            expected_visits = (
                restaurant_base_flow[restaurant_id]
                * season_multipliers[season]
                * restaurant_season_adjustment[restaurant_id][season]
                * weekend_multiplier
                * friday_multiplier
                * monday_multiplier
                * random_multiplier
            )
            visits_count = max(8, int(expected_visits))

            for _ in range(visits_count):
                customer_id = rng.choices(customer_ids, weights=customer_weights, k=1)[0]
                profile = customer_profiles[customer_id]
                visit_time = choose_visit_time()
                lines_count = rng.choices([1, 2, 3], weights=[0.58, 0.34, 0.08], k=1)[0]
                used_items = set()

                for _ in range(lines_count):
                    candidate_items = restaurant_menu_items[restaurant_id]
                    item_weights = []

                    for item in candidate_items:
                        if item["id"] in used_items:
                            item_weights.append(Decimal("0.01"))
                        elif item["is_seasonal"] and item["season"] == season:
                            item_weights.append(Decimal("3.20"))
                        elif item["is_seasonal"]:
                            item_weights.append(Decimal("0.15"))
                        else:
                            item_weights.append(Decimal("1.00"))

                    menu_item = rng.choices(candidate_items, weights=[float(weight) for weight in item_weights], k=1)[0]
                    used_items.add(menu_item["id"])

                    quantity = rng.choices([1, 2, 3], weights=[0.86, 0.12, 0.02], k=1)[0]
                    unit_price = menu_item["standard_price"]
                    cost_amount = money(menu_item["base_cost"] * quantity)
                    gross_revenue = money(unit_price * quantity)
                    promo_discount = Decimal("0.05") if weekday in (1, 2) and rng.random() < 0.18 else Decimal("0.00")
                    discount_rate = min(Decimal("0.15"), loyalty_discount[profile["loyalty_status"]] + promo_discount)
                    discount_amount = money(gross_revenue * discount_rate)
                    revenue_amount = money(gross_revenue - discount_amount)
                    profit_amount = money(revenue_amount - cost_amount)
                    preparation_time = rng.randint(6, 28)
                    service_time = rng.randint(18, 65)
                    rating_noise = rng.gauss(0, 0.28)
                    rating_load_penalty = -0.10 if weekend_multiplier > Decimal("1.00") and rng.random() < 0.35 else 0.00
                    rating = max(3.0, min(5.0, rating_base[restaurant_id] + rating_noise + rating_load_penalty))
                    rating_value = Decimal(str(round(rating, 2))).quantize(Decimal("0.01"))

                    sales.append(
                        (
                            sales_fact_id,
                            date_id(current),
                            time_id(visit_time),
                            restaurant_id,
                            menu_item["id"],
                            customer_id,
                            quantity,
                            unit_price,
                            discount_amount,
                            cost_amount,
                            revenue_amount,
                            profit_amount,
                            preparation_time,
                            service_time,
                            rating_value,
                        )
                    )
                    sales_fact_id += 1

        current += timedelta(days=1)

    conn.executemany(
        "INSERT INTO Fact_Sales VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        sales,
    )

def build_database(
    db_path: str,
    start_date: str,
    end_date: str,
    demo: bool,
) -> None:
    path = Path(db_path)

    with duckdb.connect(str(path)) as conn:
        recreate_schema(conn)
        fill_date_dimension(conn, start_date=start_date, end_date=end_date)
        fill_time_dimension(conn)

        if demo:
            fill_demo_data(conn)

        conn.execute("CHECKPOINT")

    print(f"База данных создана: {path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-12-31")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    build_database(
        db_path=args.db,
        start_date=args.start_date,
        end_date=args.end_date,
        demo=args.demo,
    )


if __name__ == "__main__":
    main()
