"""DS 15분 배정 DAG (S15P21E105-102 하이브리드, 소유: DS1).

매시 :20 :35 :50 에 ds_chain 과 같은 이미지를 MODE=assign 으로 돌린다 — 적재·전처리·
중복 제거·임베딩까지는 같고, 재군집 대신 **배정**(hannun-assign): 새 대표 기사를 기존
이슈 중심과 코사인 0.85 이상이면 그 이슈에 붙이고, 못 붙는 기사는 노이즈로 두어
다음 정시 재군집(ds_chain, :05)에 맡긴다. 배정 뒤 승계·품질·피드·내보내기는 그대로
돌아 붙은 기사가 15분 안에 registry 와 ds_output 에 반영된다.

왜 배정인가: 15분 전체 재군집은 UMAP 지도를 매번 다시 그려 경계의 작은 이슈가
쪼개졌다 붙었다 했다(주말 실측 created 40~57/실행). 배정은 기존 이슈를 건드리지
않아 그 출렁임이 없다 — 대신 새 이슈 탄생은 정시까지 최대 1시간 기다린다.

ds_chain 과 같은 flock 을 쓴다. 재군집이 길어져 겹치면 4분까지 기다렸다 양보한다.
"""

from datetime import datetime, timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

KST = pendulum.timezone("Asia/Seoul")

with DAG(
    dag_id="ds_assign",
    start_date=datetime(2026, 9, 14, tzinfo=KST),
    schedule="20,35,50 * * * *",
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    tags=["ds"],
) as dag:
    BashOperator(
        task_id="hannun_assign_chain",
        retries=0,
        execution_timeout=timedelta(minutes=20),
        bash_command=(
            "ssh -i /opt/airflow/.ssh/hannun.pem"
            " -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
            " ubuntu@host.docker.internal"
            " 'set -o pipefail; flock -w 240 /tmp/hannun_chain.lock"
            " docker run --rm -u 1000:1000 -e MODE=assign"
            " -v $HOME/gold:/data/gold -v $HOME/ds_input:/data/ds_input"
            " -v $HOME/ds_output:/data/ds_output"
            " -v $HOME/.cache/huggingface:/data/hf_cache hannun-ds"
            " 2>&1 | tee -a $HOME/hannun_chain.log'"
        ),
    )
