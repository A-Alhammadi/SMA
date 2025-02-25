# database.py

import psycopg2
import pandas as pd
from psycopg2 import pool
from config import DB_CONFIG

class DatabaseHandler:
    # Class-level connection pool
    _pool = None
    
    @classmethod
    def initialize_pool(cls, min_conn=1, max_conn=10):
        """Initialize the connection pool if it doesn't exist yet"""
        if cls._pool is None:
            try:
                cls._pool = pool.ThreadedConnectionPool(min_conn, max_conn, **DB_CONFIG)
                print(f"Connection pool initialized with {min_conn}-{max_conn} connections")
            except Exception as e:
                print(f"Error initializing connection pool: {str(e)}")
                raise
    
    def __init__(self):
        """Initialize the DatabaseHandler, creating pool if necessary"""
        self.conn = None
        # Ensure pool exists
        if DatabaseHandler._pool is None:
            DatabaseHandler.initialize_pool()
        self.connect()

    def connect(self):
        """Get a connection from the pool"""
        try:
            if self.conn is None:
                self.conn = DatabaseHandler._pool.getconn()
                print("Successfully acquired connection from pool")
        except Exception as e:
            print(f"Error getting connection from pool: {str(e)}")
            raise

    def get_historical_data(self, symbol, start_date, end_date):
        """Get historical data with precise hourly timestamps"""
        query = """
            SELECT 
                date_time,
                open_price,
                high_price,
                low_price,
                close_price,
                volume_crypto,
                volume_usd
            FROM crypto_data_hourly
            WHERE symbol = %s
              AND date_time >= %s::timestamp
              AND date_time <= %s::timestamp
            ORDER BY date_time ASC
        """
        
        try:
            print(f"\nFetching data for {symbol}:")
            print(f"Start: {start_date}")
            print(f"End: {end_date}")
            
            df = pd.read_sql_query(
                query,
                self.conn,
                params=(symbol, start_date, end_date),
                parse_dates=['date_time']
            )
            
            # Set date_time as the DataFrame index
            df.set_index('date_time', inplace=True)
            
            print(f"Fetched {len(df)} hourly records")
            if len(df) > 0:
                print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
                
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise

    def close(self):
        """Return the connection to the pool instead of closing it"""
        if self.conn:
            DatabaseHandler._pool.putconn(self.conn)
            self.conn = None
            print("Connection returned to pool.")
    
    @classmethod
    def shutdown_pool(cls):
        """Properly close all pool connections when application shuts down"""
        if cls._pool is not None:
            cls._pool.closeall()
            cls._pool = None
            print("Database connection pool closed.")