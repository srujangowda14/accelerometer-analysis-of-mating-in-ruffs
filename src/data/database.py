import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from typing import List, Optional, Dict
import logging

class AccelerometerDB:
    """Interface for the accelerometr DB"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.logger = logging.getLogger(__name__)

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def query_raw_data(self, bird_id: str,
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None,
                       chunk_size: int = 10000) -> pd.DataFrame:
        """
        Query raw accelerometer data for a specific bird
        
        Args:
            bird_id: Bird identifier
            start_time: Start timestamp
            end_time: End timestamp
            chunk_size: Number of rows per chunk for memory efficiency
        
        Returns:
            DataFrame with accelerometer readings
        """
        
        query = f"""
        SELECT timestamp, acc_x, acc_y, acc_z
        FROM accelerometer_data
        WHERE bird_id = '{bird_id}'
        """

        if start_time:
            query += f" AND timestamp >= '{start_time}'"

        if end_time:
            query += f" AND timestamp <= '{end_time}'"

        query += " ORDER by timestamp"

        #Read in chunk for memory efficiency

        chunks = []
        for chunk in pd.read_sql_query(query, self.engine, chunksize = chunk_size):
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index = True)
    
    def get_bird_ids(self) -> List[str]:
        """Get list of all bird IDs in database"""
        query = "SELECT DISTINCT bird_id FROM accelerometer_data"
        df = pd.read_sql_query(query, self.engine)
        return df['bird_id'].tolist()
    
    def get_deployment_info(self, bird_id: str) -> Dict:
        """Get deployment information for a bird"""
        query = f"""
        SELECT * FROM deployment_info
        WHERE bird_id = '{bird_id}'
        """
        df = pd.read_sql_query(query, self.engine)
        return df.to_dict('records')[0] if len(df) > 0 else {}
    
    







