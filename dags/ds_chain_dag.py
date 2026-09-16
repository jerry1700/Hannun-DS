"""DS 체인 DAG — 매시 :05 전체 재군집 (S15P21E105-101, -102, -122. 소유: DS1).

DS 파이프라인 여덟 단계 + DS2 관점·발행용 JSONL 내보내기(ds_output)를 hannun-ds
이미지로 한 번에 돌린다. 사슬 내용은 DS 소유이고 DE 는 부르기만 한다 — ds_daily
DAG 가 이 DAG 를 켜고, 끝나면 gold_issue_feed 가 gold 와 ds_output 을 병합한다.

매시 재군집인 이유(하이브리드, 티켓 102): 15분 전체 재군집은 UMAP 지도가 매번 미세하게
재배열돼 경계의 작은 이슈가 쪼개졌다 붙었다 하며 ID 를 낭비했다(주말 실측). 그래서 새
이슈 탄생·분열·병합은 매시 1회로 하고, 그 사이 15분 배정은 ds_assign DAG(같은 이미지,
MODE=assign)가 맡는다. :05 출발은 DE article_ingest(:00~:03)가 갓 내려준 입력을 받고
CPU 충돌을 피하기 위해서. ds_daily 의 04:00 트리거는 공존(멱등·직렬화로 무해).

호스트에 ssh 로 들어가 컨테이너를 돌린다. docker.sock 마운트 대신 ssh 인 이유: 소켓은
사실상 서버 관리자 권한이라(compose 의 UI 포트 주석) ubuntu 권한으로 한정하고,
backup_to_b 와 같은 검증된 키 패턴(uid 50000 마운트)을 재사용한다. flock 은 ds_assign
과의 동시 기동 방지 — 사슬 자체는 멱등이라 겹쳐도 데이터는 안전하다.
"""

from datetime import datetime, timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

KST = pendulum.timezone("Asia/Seoul")

with DAG(
    dag_id="ds_chain",
    start_date=datetime(2026, 9, 10, tzinfo=KST),
    schedule="5 * * * *",
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    tags=["ds"],
) as dag:
    BashOperator(
        # 이름의 daily 는 일일 배치 시절 것 — 바꾸면 Airflow 가 이력을 새 태스크로 취급해 그대로 둔다
        task_id="hannun_daily_chain",
        # 다음 정시 실행이 자연 재시도다 — 별도 재시도는 잠금 경합 소음만 만든다.
        # 타임아웃은 걸린 실행이 뒤 실행들을 오래 막지 않게 30분. flock 은 ds_assign
        # 과 같은 잠금을 쓰고 10분까지 기다린다(배정이 끝나기를 기다렸다 이어 돎).
        # 명령은 ds_assign_dag.py 와 MODE 만 다르다 — 고칠 때 둘을 같이 고친다
        retries=0,
        execution_timeout=timedelta(minutes=30),
        bash_command=(
            "ssh -i /opt/airflow/.ssh/hannun.pem"
            " -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
            " ubuntu@host.docker.internal"
            " 'set -o pipefail; flock -w 600 /tmp/hannun_chain.lock"
            " docker run --rm -u 1000:1000"
            " -v $HOME/gold:/data/gold -v $HOME/ds_input:/data/ds_input"
            " -v $HOME/ds_output:/data/ds_output"
            " -v $HOME/.cache/huggingface:/data/hf_cache hannun-ds"
            " 2>&1 | tee -a $HOME/hannun_chain.log'"
        ),
    )
