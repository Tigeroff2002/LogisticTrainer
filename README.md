Запуск сервера: uvicorn main:app --reload

По адресу http://127.0.0.1:8000/docs# доступен Swagger

Основной акцент я планировал сделать тут на уточнение времени пути
От OSRM GEO API я получаю некоторый expected_time - но он может быть уточнен исходя из разреза времени
А именно параметров time_of_day_directory_id (приоритет 1), day_of_week (приоритет 2), month_of_year (приоритет 3)

Этот сервис может:
1) Запускать тренировку (на основании данных таблицы users_history_directory), получать модель и сохранять ее в object storage (minio) - его нужно в Docker развернуть
При повторной тренировке - последней сохраняется новое состояние обученной модели
2) Есть REST endpoint для вызова запроса к обученной модели - я думаю давать ей параметры из файла prediction_request_to_be (он пока не используется, используется prediction_request)
И на выходе получать уточненное время (уточненное в первую очередь категориальными признаками, связанными с разрезом времени)
И так же я думал насчет того чтобы персонализировать время предсказания - то есть еще ориентироваться при наличии данных на user_id

Таблица Postgre SQL, с которой происходит чтение данных, представляет вид:

CREATE TABLE users (
    id bigint GENERATED ALWAYS AS IDENTITY,
    email text NOT NULL,
    role role NOT NULL,
    CONSTRAINT pk_users PRIMARY KEY (id));

CREATE TYPE area_type AS ENUM (
    'small', --area lesser than 30m
    'medium',  -- area on range [30, 100]m
    'complex'); -- area on range over 100m

# Таблица с избранными областями 
# (концентрированные области разного радиуса с точками, в которых пользователи начинали/заканчивали маршруты)
CREATE TABLE favorite_areas(
    id bigint GENERATED ALWAYS AS IDENTITY,
    user_id bigint NOT NULL,
    area_type area_type NOT NULL,
    width double precision NOT NULL,
    height double precision NOT NULL,
    CONSTRAINT pk_favorite_areas PRIMARY KEY (id),
    CONSTRAINT fk_favorite_areas_user_id
        FOREIGN KEY (user_id)
        REFERENCES users (id) ON DELETE CASCADE);

# Справочник с временными рэнджами в течении дня
CREATE TABLE time_of_day_directory(
    id bigint GENERATED ALWAYS AS IDENTITY,
    start_range INTERVAL NOT NULL,
    end_range INTERVAL NOT NULL,
	CONSTRAINT pk_time_of_day_directory PRIMARY KEY (id));

INSERT INTO time_of_day_directory (start_range, end_range) VALUES ('1 hours'::interval, '6 hours'::interval);
INSERT INTO time_of_day_directory (start_range, end_range) VALUES ('6 hours'::interval, '9 hours'::interval);
INSERT INTO time_of_day_directory (start_range, end_range) VALUES ('9 hours'::interval, '12 hours'::interval);
INSERT INTO time_of_day_directory (start_range, end_range) VALUES ('12 hours'::interval, '14 hours'::interval);
INSERT INTO time_of_day_directory (start_range, end_range) VALUES ('14 hours'::interval, '16 hours 30 minutes'::interval);
INSERT INTO time_of_day_directory (start_range, end_range) VALUES ('16 hours 30 minutes'::interval, '19 hours'::interval);
INSERT INTO time_of_day_directory (start_range, end_range) VALUES ('19 hours'::interval, '22 hours'::interval);
INSERT INTO time_of_day_directory (start_range, end_range) VALUES ('22 hours'::interval, '1 hours'::interval);

CREATE TYPE day_of_week AS ENUM (
    'monday', 
    'tuesday', 
    'wednesday', 
    'thursday', 
    'friday', 
    'saturday', 
    'sunday');

CREATE TYPE month_of_year AS ENUM (
    'january',
    'february',
    'march',
    'april',
    'may',
    'june',
    'july',
    'august',
    'september',
    'october',
    'november',
    'december');

# Основная таблица, наполяемая данными при финализации маршрутов
CREATE TABLE users_history_directory (
    id bigint GENERATED ALWAYS AS IDENTITY,
    user_id bigint NOT NULL,
    start_fav_area_id bigint NOT NULL, // если сделать join в favorite_areas можно получать координату центра и радиус области
    end_fav_area_id bigint NOT NULL,
    month_of_year month_of_year NOT NULL,
    time_of_day_directory_id bigint NOT NULL,
    day_of_week day_of_week NOT NULL,
    expected_duration interval NOT NULL,
    duration interval NOT NULL,
    CONSTRAINT pk_users_history_directory PRIMARY KEY (id),
    CONSTRAINT fk_users_history_directory
        FOREIGN KEY (user_id)
        REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT fk_users_history_directory_start_fav_area_id 
        FOREIGN KEY (start_fav_area_id) 
        REFERENCES favorite_areas (id) ON DELETE CASCADE,
    CONSTRAINT fk_users_history_directory_end_fav_area_id 
        FOREIGN KEY (end_fav_area_id) 
        REFERENCES favorite_areas (id) ON DELETE CASCADE,
    CONSTRAINT fk_users_history_directory_time_of_day_directory_id 
        FOREIGN KEY (time_of_day_directory_id)
        REFERENCES time_of_day_directory (id));

Там я кинул dump.sql файл моей БД локальной - его можно применить, там все данные сохранены

S C:\Program Files\PostgreSQL\17\bin> .\psql -U postgres -d logistic_salesman_db -f dump.sql