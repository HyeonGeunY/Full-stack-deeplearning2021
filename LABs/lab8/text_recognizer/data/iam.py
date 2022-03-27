import sys
sys.path.append("../..")

from pathlib import Path
from typing import Dict, List
import argparse
import os
import xml.etree.ElementTree as ElementTree
import zipfile

from boltons.cacheutils import cachedproperty
import toml

from text_recognizer.data.base_data_module import BaseDataModule, _download_raw_dataset, load_and_print_info


RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "iam"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "iam"
EXTRACTED_DATASET_DIRNAME = DL_DATA_DIRNAME / "iamdb"

DOWNSAMPLE_FACTOR = 2
LINE_REGION_PADDING = 16

class IAM(BaseDataModule):
    """
    "The IAM Lines dataset, first published at the ICDAR 1999, contains forms of unconstrained handwritten text,
    which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels.
    From http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

    The data split we will use is
    IAM lines Large Writer Independent Text Line Recognition Task (lwitlrt): 9,862 text lines.
        The validation set has been merged into the train set.
        The train set has 7,101 lines from 326 writers.
        The test set has 1,861 lines from 128 writers.
        The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.
    """
    
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        # meta 데이터에는 훈련시 사용한 test 데이터의 인덱스 정보를 담아두어 재생성이 용의하게 한다.
        self.metadata = toml.load(METADATA_FILENAME)
    
    def prepare_data(self, *args, **kwargs) -> None:
        if self.xml_filenames:
            return
        filename = _download_raw_dataset(self.metadata, DL_DATA_DIRNAME)
        _extract_raw_dataset(filename, DL_DATA_DIRNAME)
        
    @property
    def xml_filenames(self):
        return list((EXTRACTED_DATASET_DIRNAME / "xml").glob("*.xml"))
    
    @property
    def form_filenames(self):
        return list((EXTRACTED_DATASET_DIRNAME / "forms").glob("*.jpg"))
    
    @property
    def form_filenames_by_id(self):
        return {filename.stem: filename for filename in self.form_filenames} 
                # stem : suffix가 없는 마지막 경로
    
    @property
    def split_by_id(self):
        return {
            filename.stem: "test" if filename.stem in self.metadata["test_ids"] else "trainval" for filename in self.form_filenames
        } # filename 별로 test와 train 구분
    
    # cachedproperty 처음 호출 시 연산 수행, 인스턴스에 값을 저장하여 연산 중복 방지
    @cachedproperty
    def line_strings_by_id(self):
        """ Return a dict from name of IAM form to a list of line texts in it."""
        return {filename.stem: _get_line_strings_from_xml_file(filename) for filename in self.xml_filenames} # 이미지 파일에 있는 단어들의 리스트 반환
    
    @cachedproperty
    def line_regions_by_id(self):
        return {filename.stem: _get_line_regions_from_xml_file(filename) for filename in self.xml_filenames} 
    
    
    def __repr__(self):
        """Print data info"""
        return "IAM Dataset\n" f"Num total: {len(self.xml_filenames)}\nNum test: {len(self.metadata['test_ids'])}\n"

def _extract_raw_dataset(filename: Path, dirname: Path) -> None:
    """
    zip 파일 풀고 원래 디렉토리로 복귀
    """
    print("Extracting IAM data")
    curdir = os.getcwd()
    os.chdir(dirname)
    with zipfile.ZipFile(filename, "r") as zip_file:
        zip_file.extractall()
    os.chdir(curdir)
    

def _get_line_strings_from_xml_file(filename: str) -> List[str]:
    """xml 형식의 annotation 파일에서 문장(line)단위로 문자를 추출하여 반환한다.
        &quot;은 " 로 변환한다.
        
        Return
        -------
        한 파일 내에 있는 line 별 문자 리스트의 리스트
        """
    # xml annotation 파일의 root 위치를 가져온다.
    xml_root_element = ElementTree.parse(filename).getroot()
    
    # xml 파일에서 <line>을 모두 찾아서 반환한다.
    xml_line_elements = xml_root_element.findall("handwritten-part/line")
    
    # <line>안에 존재하는 <text> 어트리뷰트의 &quot을 "로 대체하여 저장한다.
    # 결과물: 페이지 내에 있는 모든 라인 별 텍스트 리스트
    return [el.attrib["text"].replace("&quot;", '"') for el in xml_line_elements] # 단어 정보


def _get_line_regions_from_xml_file(filename: str) -> List[Dict[str, int]]:
    """Get the line region dict for each line."""
    xml_root_element = ElementTree.parse(filename).getroot()
    xml_line_elements = xml_root_element.findall("handwritten-part/line")
    return [_get_line_region_from_xml_element(el) for el in xml_line_elements]
    

def _get_line_region_from_xml_element(xml_line) -> Dict[str, int]:
    """
    
    xml 파일의 <line> </line>에서 cmp(x, y) 정보를 가져와 문장 단위로 이미지 내부에 존재하는 위치 정보를 추출한다.

    현재 사용하는 iam 데이터의 어노테이션 파일의 경우 단어 단위(<word><cmp>...</cmlp></word>)로 위치 정보가 표시되어 있다. 
    따라서 문장의 첫 단어와 끝단어의 위치정보를 통해 문장 위치 정보를 가져온 후 목적에 맞게 downsampling한 값을 반환한다.
    앞뒤 부분이 잘리지 않게 적절한 padding 영역을 추가한다.
    
    Parameters
    -----------
    xml_line
        xml element that has x, y, width and height attributes
    """
    
    word_elements = xml_line.findall("word/cmp")
    x1s = [int(el.attrib["x"]) for el in word_elements]
    y1s = [int(el.attrib["y"]) for el in word_elements]
    x2s = [int(el.attrib["x"]) + int(el.attrib["width"]) for el in word_elements]
    y2s = [int(el.attrib["y"]) + int(el.attrib["height"]) for el in word_elements]
    
    # 이미지 내 문장(line)의 글자가 존재하는 영역 반환
    # downsample_factor : 메모리 절약을 위해 이미지 크기 줄인 것을 반영
    # line_region_padding : 패딩영역
    return {
        "x1": min(x1s) // DOWNSAMPLE_FACTOR - LINE_REGION_PADDING,
        "y1": min(y1s) // DOWNSAMPLE_FACTOR - LINE_REGION_PADDING,
        "x2": max(x2s) // DOWNSAMPLE_FACTOR + LINE_REGION_PADDING,
        "y2": max(y2s) // DOWNSAMPLE_FACTOR + LINE_REGION_PADDING,
    }
    

if __name__ == "__main__":
    load_and_print_info(IAM)