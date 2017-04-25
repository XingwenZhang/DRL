import os
import shutil
import THORConfig as config
import THOROfflineEnv

if __name__ == '__main__':
	if os.path.exists(config.env_db_folder):
		shutil.rmtree(config.env_db_folder)
	os.mkdir(config.env_db_folder)
	db = THOROfflineEnv.EnvSim()
	db.build()