import openxlab
openxlab.login(ak='', sk='') # 进行登录，输入对应的AK/SK，可在个人中心查看AK/SK

from openxlab.dataset import info
info(dataset_repo='OpenDataLab/VISOR') #数据集信息查看

# from openxlab.dataset import get
# get(dataset_repo='OpenDataLab/VISOR', target_path='/path/to/local/folder/') # 数据集下载

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/VISOR',source_path='/raw/VISOR.tar.gz', target_path='/data/epic_kitchen') #数据集文件下载