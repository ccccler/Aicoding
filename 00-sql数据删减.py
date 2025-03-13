import pymysql
import sqlite3
import psycopg2
import sys

def delete_all_data_from_table(db_type, connection_params):
    """
    从指定数据库中删除preprocessed_data表的所有数据
    
    参数:
        db_type: 数据库类型 ('mysql', 'sqlite', 'postgresql')
        connection_params: 连接参数字典
    """
    try:
        if db_type == 'mysql':
            # MySQL连接
            conn = pymysql.connect(
                host=connection_params.get('host', 'localhost'),
                user=connection_params.get('user', 'root'),
                password=connection_params.get('password', ''),
                database=connection_params.get('database', '')
            )
        elif db_type == 'sqlite':
            # SQLite连接
            conn = sqlite3.connect(connection_params.get('database', 'database.db'))
        elif db_type == 'postgresql':
            # PostgreSQL连接
            conn = psycopg2.connect(
                host=connection_params.get('host', 'localhost'),
                user=connection_params.get('user', 'postgres'),
                password=connection_params.get('password', ''),
                dbname=connection_params.get('database', '')
            )
        else:
            print(f"不支持的数据库类型: {db_type}")
            return False
        
        cursor = conn.cursor()
        
        # 执行DELETE语句删除表中所有数据
        cursor.execute("DELETE FROM preprocessed_data")
        
        # 提交事务
        conn.commit()
        
        # 输出删除的行数（如果数据库支持）
        if db_type != 'sqlite':  # SQLite不支持rowcount
            print(f"已删除 {cursor.rowcount} 行数据")
        else:
            print("数据已成功删除")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"删除数据时出错: {e}")
        return False

if __name__ == "__main__":
    # 示例用法
    db_type = input("请输入数据库类型 (mysql/sqlite/postgresql): ").lower()
    
    connection_params = {}
    
    if db_type == 'sqlite':
        connection_params['database'] = input("请输入数据库文件路径: ")
    else:
        connection_params['host'] = input("请输入主机地址 (默认localhost): ") or 'localhost'
        connection_params['user'] = input("请输入用户名: ")
        connection_params['password'] = input("请输入密码: ")
        connection_params['database'] = input("请输入数据库名: ")
    
    success = delete_all_data_from_table(db_type, connection_params)
    
    if success:
        print("操作完成：preprocessed_data表中的所有数据已被删除")
    else:
        print("操作失败：无法删除数据")
