python3 -u NMTF-LTM.py "$1";
python3 -u evaluate_clustering.py "$1";
python3 -u evaluate_topics_nmtf.py "$1";
python3 -u IR.py "$1";
python3 -u get_result.py "$1";
