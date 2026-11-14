import os
import sys
import yaml
from pyspark.sql import SparkSession


def init_spark_environment():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    hadoop_home = r"C:\hadoop"
    if os.path.exists(hadoop_home):
        os.environ["HADOOP_HOME"] = hadoop_home
        bin_path = os.path.join(hadoop_home, "bin")
        if bin_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")


def get_spark_session(app_name="FinancialFraudBigData", config_path="config.yaml"):

    init_spark_environment()

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        master = cfg.get("spark", {}).get("master", "local[*]")
        driver_mem = cfg.get("spark", {}).get("driver_memory", "6g")
        shuffle_parts = str(cfg.get("spark", {}).get("sql_shuffle_partitions", 8))
        parallelism = str(cfg.get("spark", {}).get("default_parallelism", 8))
    else:
        master = "local[*]"
        driver_mem = "6g"
        shuffle_parts = "8"
        parallelism = "8"

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_mem)
        .config("spark.sql.shuffle.partitions", shuffle_parts)
        .config("spark.default.parallelism", parallelism)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark
