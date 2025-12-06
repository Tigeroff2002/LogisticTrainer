import asyncpg
import pandas as pd
from datetime import datetime
import logging
from app.config import Config

class DatabaseService:
    def __init__(self, config: Config):
        self.config = config
        self.connection: asyncpg.Connection = None
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== DATABASE SERVICE INITIALIZED ===")

    async def connect(self):
        """Установка соединения с PostgreSQL"""
        try:
            self.connection = await asyncpg.connect(
                host=self.config.database.host,
                port=self.config.database.port,
                database=self.config.database.database,
                user=self.config.database.user,
                password=self.config.database.password
            )
            self.logger.info("Successfully connected to PostgreSQL")
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self):
        """Закрытие соединения"""
        if self.connection:
            await self.connection.close()
            self.logger.info("Disconnected from PostgreSQL")

    async def get_training_data(self) -> pd.DataFrame:
        """Получение исторических данных для обучения"""
        self.logger.info("Starting to fetch training data from PostgreSQL")
        
        query = """
        SELECT 
            uh.id,
            user_id,
            start_fav_area_id,
            end_fav_area_id,
            month_of_year,
            CONCAT(tod.start_range, ':', tod.end_range) as time_of_day,
            day_of_week,
            EXTRACT(EPOCH FROM duration) as duration_seconds
        FROM users_history_directory uh
        JOIN time_of_day_directory tod ON uh.time_of_day_directory_id = tod.id
        ORDER BY uh.id ASC
        LIMIT 100000
        """
        
        try:
            # Выполняем запрос
            records = await self.connection.fetch(query)
            self.logger.info(f"Fetched {len(records)} historical records from database")
            
            # Конвертируем в DataFrame
            df = pd.DataFrame(
                [dict(record) for record in records],
                columns=['user_id', 'start_fav_area_id', 'end_fav_area_id', 'month_of_year', 
                        'time_of_day', 'day_of_week', 'number_of_rides', 'duration_seconds']
            )
            
            self.logger.info(f"Training data shape: {df.shape}")
            self.logger.info(f"Columns: {df.columns.tolist()}")
            self.logger.info(f"Sample data:\n{df.head()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching training data: {e}")
            raise