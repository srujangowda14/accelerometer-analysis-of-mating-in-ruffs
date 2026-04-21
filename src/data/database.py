import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from typing import List, Optional, Dict
import logging


class AccelerometerDB:
    """Interface for the accelerometer database"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect table name
        self.table_name = self._detect_table_name()
        self.logger.info(f"Using table: {self.table_name}")
    
    def _detect_table_name(self):
        """Detect the correct table name from the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        
        # Try common table names in order of preference
        for name in ['accelerometer_data', 'acc', 'accelerometer', 'data']:
            if name in tables:
                conn.close()
                return name
        
        conn.close()
        
        # If no standard name found, use first non-system table
        non_system_tables = [t for t in tables if not t.startswith('sqlite_')]
        if non_system_tables:
            return non_system_tables[0]
        
        raise ValueError(f"No suitable data table found. Tables: {tables}")
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def _get_acc_columns(self) -> List[str]:
        """Return column names for the primary accelerometer table."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {self.table_name} LIMIT 1")
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return columns

    def _has_table(self, table_name: str) -> bool:
        """Check whether a table exists in the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def _resolve_recording_pk(self, recording_id: str) -> Optional[int]:
        """Resolve an external recording identifier to the internal DB primary key."""
        if not self._has_table('recordings'):
            return None

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM recordings WHERE recording_id = ?",
            (recording_id,),
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    
    def query_raw_data(self, bird_id: str,
                      start_time: Optional[str] = None,
                      end_time: Optional[str] = None,
                      limit: Optional[int] = None,
                      chunk_size: int = 10000) -> pd.DataFrame:
        """
        Query raw accelerometer data for a specific bird
        
        Args:
            bird_id: Bird identifier
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of rows to return
            chunk_size: Number of rows per chunk for memory efficiency
            
        Returns:
            DataFrame with accelerometer readings
        """
        columns = self._get_acc_columns()

        # Map common column name variations
        col_mapping = self._detect_column_names(columns)

        params = {'entity_id': bird_id}

        # Prefer joining through the recordings table when available because the
        # accelerometer table often stores only an internal numeric key.
        if (
            self.table_name == 'acc'
            and self._has_table('recordings')
            and 'recording_id' in columns
        ):
            recording_pk = self._resolve_recording_pk(str(bird_id))
            if recording_pk is None:
                return pd.DataFrame(columns=['timestamp', 'acc_x', 'acc_y', 'acc_z'])
            params = {'entity_id': recording_pk}
            query = f"""
            SELECT a.recording_id AS recording_id,
                   a.{col_mapping['timestamp']} AS timestamp,
                   a.{col_mapping['acc_x']} AS acc_x,
                   a.{col_mapping['acc_y']} AS acc_y,
                   a.{col_mapping['acc_z']} AS acc_z
            FROM {self.table_name} a
            WHERE a.recording_id = :entity_id
            """
        else:
            query = f"""
            SELECT {col_mapping['bird_id']} AS bird_id,
                   {col_mapping['timestamp']} AS timestamp,
                   {col_mapping['acc_x']} AS acc_x,
                   {col_mapping['acc_y']} AS acc_y,
                   {col_mapping['acc_z']} AS acc_z
            FROM {self.table_name}
            WHERE {col_mapping['bird_id']} = :entity_id
            """

        if start_time:
            query += " AND timestamp >= :start_time"
            params['start_time'] = start_time
        if end_time:
            query += " AND timestamp <= :end_time"
            params['end_time'] = end_time

        query += " ORDER BY timestamp"

        if limit:
            query += " LIMIT :limit"
            params['limit'] = limit

        # Read in chunks for memory efficiency
        chunks = []
        for chunk in pd.read_sql_query(
            query,
            self.engine,
            params=params,
            chunksize=chunk_size,
        ):
            chunks.append(chunk)
        
        if not chunks:
            return pd.DataFrame(columns=['timestamp', 'acc_x', 'acc_y', 'acc_z'])
        
        return pd.concat(chunks, ignore_index=True)
    
    def _detect_column_names(
        self,
        columns: List[str],
        require_bird_id: bool = True,
    ) -> Dict[str, str]:
        """Detect column name variations"""
        col_map = {}
        
        # Detect bird_id column
        if require_bird_id:
            bird_id_options = ['bird_id', 'individual_id', 'tag_id', 'animal_id', 'id', 
                              'recording_id', 'device_id', 'logger_id']
            for opt in bird_id_options:
                if opt in columns:
                    col_map['bird_id'] = opt
                    break
            if 'bird_id' not in col_map:
                raise ValueError(f"Could not find bird ID column. Columns: {columns}")
        
        # Detect timestamp column
        timestamp_options = ['timestamp', 'time', 'datetime', 'date_time', 'dt']
        for opt in timestamp_options:
            if opt in columns:
                col_map['timestamp'] = opt
                break
        if 'timestamp' not in col_map:
            raise ValueError(f"Could not find timestamp column. Columns: {columns}")
        
        # Detect accelerometer columns (handle both lowercase and camelCase)
        for axis in ['x', 'y', 'z']:
            axis_options = [
                f'acc_{axis}',           # acc_x
                f'acceleration_{axis}',  # acceleration_x
                axis,                     # x
                f'a{axis}',              # ax
                f'acc{axis.upper()}',    # accX (your format!)
                f'acc{axis}',            # accx
            ]
            for opt in axis_options:
                if opt in columns:
                    col_map[f'acc_{axis}'] = opt
                    break
            if f'acc_{axis}' not in col_map:
                raise ValueError(f"Could not find {axis}-axis column. Columns: {columns}")
        
        return col_map
    
    def get_bird_ids(self) -> List[str]:
        """Get list of all bird IDs in database"""
        if self.table_name == 'acc' and self._has_table('recordings'):
            query = "SELECT recording_id FROM recordings ORDER BY recording_id"
            df = pd.read_sql_query(query, self.engine)
            return df['recording_id'].astype(str).tolist()

        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Detect bird_id column
        columns = self._get_acc_columns()
        col_mapping = self._detect_column_names(
            columns,
            require_bird_id=not (
                self.table_name == 'acc'
                and self._has_table('recordings')
                and 'recording_id' in columns
            ),
        )
        
        # Query distinct bird IDs
        query = f"SELECT DISTINCT {col_mapping['bird_id']} FROM {self.table_name}"
        cursor.execute(query)
        bird_ids = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return bird_ids
    
    def get_deployment_info(self, bird_id: str) -> Dict:
        """Get deployment information for a bird"""
        # Check if deployment_info table exists
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='deployment_info'")
        
        if cursor.fetchone():
            query = f"""
            SELECT * FROM deployment_info
            WHERE bird_id = '{bird_id}'
            """
            df = pd.read_sql_query(query, self.engine)
            conn.close()
            return df.to_dict('records')[0] if len(df) > 0 else {}
        else:
            conn.close()
            return {}
    
    def get_data_summary(self, bird_id: str) -> Dict:
        """
        Get summary statistics for a bird's data
        
        Args:
            bird_id: Bird identifier
            
        Returns:
            Dictionary with summary statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Detect column names
        cursor.execute(f"SELECT * FROM {self.table_name} LIMIT 1")
        columns = [desc[0] for desc in cursor.description]
        col_mapping = self._detect_column_names(columns)
        
        query = f"""
        SELECT 
            COUNT(*) as n_samples,
            MIN({col_mapping['timestamp']}) as start_time,
            MAX({col_mapping['timestamp']}) as end_time
        FROM {self.table_name}
        WHERE {col_mapping['bird_id']} = '{bird_id}'
        """
        
        df = pd.read_sql_query(query, self.engine)
        conn.close()
        
        return df.to_dict('records')[0] if len(df) > 0 else {}
