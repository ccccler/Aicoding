import sqlite3
import pandas as pd



# 删除数据示例
def delete_data(db_path: str, table_name: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    delete_query = f"DELETE FROM {table_name}"
    cursor.execute(delete_query)
    conn.commit()  # 提交更改

# 数据导出
def export_data(db_path: str, output_path: str, table_name: str):
    conn= sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)

    # 导出到XLSX文件
    df.to_excel(output_path, index=False)

    print(f"数据已成功导出到 {output_path}")

# 关闭连接
    conn.close()

if __name__ == "__main__":

    db_path = "nrfx/03-analy/01-themes_results.db"
    output_path = "nrfx/03-analy/01-themes_results_data.xlsx"
    table_name = "themes"

    export_data(db_path, output_path, table_name)
    # delete_data(db_path, table_name)

    
    