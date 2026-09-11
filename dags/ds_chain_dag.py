"""DS 일일 사슬 DAG (S15P21E105-101, -122).

DS 파이프라인 여덟 단계 + DS2 관점·발행용 JSONL 내보내기(ds_output)를 hannun-ds
이미지로 한 번에 돌린다. 사슬 내용은 DS 소유이고 DE 는 부르기만 한다 — ds_daily
DAG 가 이 DAG 를 켜고, 끝나면 gold_issue_feed 가 gold 와 ds_output 을 병합한다.

스스로 예약하지 않는다. 창을 부르는 쪽이 정하게 해서 언제 눌렀느냐로 산출물 창이
달라지는 일을 막는다.

호스트에 ssh 로 들어가 컨테이너를 돌린다. docker.sock 마운트 대신 ssh 인 이유: 소켓은
사실상 서버 관리자 권한이라(compose 의 UI 포트 주석) ubuntu 권한으로 한정하고,
backup_to_b 와 같은 검증된 키 패턴(uid 50000 마운트)을 재사용한다. flock 은 cron
폴백과의 동시 기동 방지 — 사슬 자체는 멱등이라 겹쳐도 데이터는 안전하다.
"""

from datetime import datetime, timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

KST = pendulum.timezone("Asia/Seoul")

with DAG(
    dag_id="ds_chain",
    start_date=datetime(2026, 9, 10, tzinfo=KST),
    schedule=None,  # ds_export DAG 가 트리거한다
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    tags=["ds"],
) as dag:
    BashOperator(
        task_id="hannun_daily_chain",
        retries=1,
        retry_delay=timedelta(minutes=20),
        execution_timeout=timedelta(hours=2),
        bash_command=(
            "ssh -i /opt/airflow/.ssh/hannun.pem"
            " -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
            " ubuntu@host.docker.internal"
            " 'set -o pipefail; flock -n /tmp/hannun_chain.lock"
            " docker run --rm -u 1000:1000"
            " -v $HOME/gold:/data/gold -v $HOME/ds_input:/data/ds_input"
            " -v $HOME/ds_output:/data/ds_output"
            " -v $HOME/.cache/huggingface:/data/hf_cache hannun-ds"
            " 2>&1 | tee -a $HOME/hannun_chain.log'"
        ),
    )
