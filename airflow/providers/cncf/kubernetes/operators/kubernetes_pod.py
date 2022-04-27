from airflow import DAG
from datetime import datetime, timedelta
# from airflow.kubernetes.secret import Secret
from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from kubernetes.client import models as k8s  # you should write this sentence when you could use volume, etc
from typing import Sequence
import json
# from airflow.operators.dagrun_operator import TriggerDagRunOperator


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime.utcnow(),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'render_template_as_native_obj': True,
    'provide_context': True,
}
dag = DAG(
    'ml_run_k8s', default_args=default_args, schedule_interval=None)

start = DummyOperator(task_id='start', dag=dag)

# volume_mount = k8s.V1VolumeMount(
#     name='test-pvc', mount_path='/workspace', sub_path=None, read_only=False
# )

# volume = k8s.V1Volume(
#     name='test-pvc',
#     # persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name='test-volume', read_only=False),
#     host_path=k8s.V1HostPathVolumeSource(path='/workspace'),
# )

configmaps = [
    k8s.V1EnvFromSource(config_map_ref=k8s.V1ConfigMapEnvSource(name='airflow-test-1')),
]


def get_command_name(experiment_process_type):
    command_dict = {  # TODO 이 참에 이거 전부 통일하면 안될까? ml_ + process_type
        'parse': 'ml_parse',
        'labeling': 'lb_tagtext',
        'lb_predict': 'lb_predict',
        'preprocess': 'ml_preprocess',
        'optuna': 'ml_optuna',
        'modelstat': 'ml_modelstat',
        'deploy': 'ml_deploy',
        'predict': 'ml_predict',
        'ensemble': 'ml_ensemble',
        'cluster': 'cl_run',
        'cl_predict': 'cl_predict',
        'dataset_eda': 'ml_dataset_eda',
        'optuna_monitor': 'ml_optuna_monitor',
        'ensemble_monitor': 'ml_ensemble_monitor',
        'None': 'None',
    }
    return command_dict[experiment_process_type]


def make_uuid():
    import uuid
    return str(uuid.uuid4()).replace('-', '')


def make_accutuning_docker_command(django_command, experiment_id, container_uuid, execute_range, experiment_process_type, proceed_next, triggered_dag_id, triggered_dag_run_id, airflow_dag_call_id, targets):
    command = f'''/code/manage.py {django_command}
--experiment={experiment_id} --uuid={container_uuid} --execute_range={execute_range}
--experiment_process_type={experiment_process_type} --proceed_next={proceed_next}
--triggered_dag_id={triggered_dag_id} --triggered_dag_run_id={triggered_dag_run_id}
--airflow_dag_call_id={airflow_dag_call_id} '''
    return command + '\n'.join([f'--{k}={v}' for (k, v) in targets.items() if v])


def make_parameters(**kwargs):
    experiment_id = kwargs['dag_run'].conf.get('ACCUTUNING_EXPERIMENT_ID')
    experiment_process_type = kwargs['dag_run'].conf.get('experiment_process_type')
    proceed_next = kwargs['dag_run'].conf.get('proceed_next')
    container_uuid = make_uuid()
    django_command = get_command_name(experiment_process_type)
    targets = dict(
        target_dataset=kwargs['dag_run'].conf.get('target_dataset'),
        target_dataset_eda=kwargs['dag_run'].conf.get('target_dataset_eda'),
        target_prediction=kwargs['dag_run'].conf.get('target_prediction'),
        target_model_base=kwargs['dag_run'].conf.get('target_model_base'),
        target_deployment=kwargs['dag_run'].conf.get('target_deployment'),
        target_source=kwargs['dag_run'].conf.get('target_source'),
    )
    triggered_dag_id = 'ml_run_k8s'
    triggered_dag_run_id = kwargs['dag_run'].run_id
    airflow_dag_call_id = kwargs['dag_run'].conf['airflow_dag_call_id']

    docker_command_before = make_accutuning_docker_command(django_command, experiment_id, container_uuid, 'before', experiment_process_type, proceed_next, triggered_dag_id, triggered_dag_run_id, airflow_dag_call_id, targets)
    docker_command_after = make_accutuning_docker_command(django_command, experiment_id, container_uuid, 'after', experiment_process_type, proceed_next, triggered_dag_id, triggered_dag_run_id, airflow_dag_call_id, targets)

    kwargs['task_instance'].xcom_push(key='before_command', value=docker_command_before)
    kwargs['task_instance'].xcom_push(key='after_command', value=docker_command_after)


def make_worker_env(**kwargs):
    workspace_path = kwargs['task_instance'].xcom_pull(task_ids='before_worker', key='return_value')["worker_workspace"]

    worker_env_vars_str = kwargs['dag_run'].conf.get('worker_env_vars')

    env_dict = json.loads(worker_env_vars_str)

    env_dict['ACCUTUNING_WORKSPACE'] = workspace_path
    env_dict['ACCUTUNING_LOG_LEVEL'] = kwargs['dag_run'].conf['ACCUTUNING_LOG_LEVEL']
    env_dict['ACCUTUNING_USE_LABELER'] = kwargs['dag_run'].conf['ACCUTUNING_USE_LABELER']
    env_dict['ACCUTUNING_USE_CLUSTERING'] = kwargs['dag_run'].conf['ACCUTUNING_USE_CLUSTERING']
    env_dict['DJANGO_SETTINGS_MODULE'] = kwargs['dag_run'].conf['DJANGO_SETTINGS_MODULE']

    kwargs['task_instance'].xcom_push(key='ACCUTUNING_WORKER_WORKSPACE', value=workspace_path)
    kwargs['task_instance'].xcom_push(key='worker_env_vars', value=env_dict)


def make_env_var():
    env_dict = {
        'ACCUTUNING_WORKSPACE': '{{dag_run.conf.ACCUTUNING_WORKSPACE}}',
        'ACCUTUNING_LOG_LEVEL': '{{dag_run.conf.ACCUTUNING_LOG_LEVEL}}',
        'ACCUTUNING_USE_LABELER': '{{dag_run.conf.ACCUTUNING_USE_LABELER}}',
        'ACCUTUNING_USE_CLUSTERING': '{{dag_run.conf.ACCUTUNING_USE_CLUSTERING}}',
        'DJANGO_SETTINGS_MODULE': '{{dag_run.conf.DJANGO_SETTINGS_MODULE}}'
    }
    return env_dict


def make_namespace():
    print("{{dag_run.conf.ACCUTUNING_NAMESPACE}}")
    return '{{dag_run.conf.ACCUTUNING_NAMESPACE}}'


parameters = PythonOperator(task_id='make_parameters', python_callable=make_parameters, dag=dag)


def set_default_volumn_mount(self, *args, **kwargs):
    volume_mounts = k8s.V1VolumeMount(
        name=kwargs['context']['dag_run'].conf.get('ACCUTUNING_PVC_NAME'),
        mount_path=kwargs['context']['dag_run'].conf.get('ACCUTUNING_WORKSPACE'),
        sub_path=None, read_only=False
    )
    self.volume_mounts = [volume_mounts]

    volumes = k8s.V1Volume(
        name=kwargs['context']['dag_run'].conf.get('ACCUTUNING_PVC_NAME'),
        # persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name='test-volume', read_only=False),
        host_path=k8s.V1HostPathVolumeSource(path=kwargs['context']['dag_run'].conf.get('ACCUTUNING_WORKSPACE')),
    )
    self.volumes = [volumes]


class KubernetesPodExPreOperator(KubernetesPodOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pre_execute(self, *args, **kwargs):
        # self.namespace = kwargs['context']['dag_run'].conf.get('ACCUTUNING_NAMESPACE')
        set_default_volumn_mount(self, *args, **kwargs)
        self.arguments = kwargs['context']['task_instance'].xcom_pull(
            task_ids='make_parameters', key='before_command').split()
        return super().pre_execute(*args, **kwargs)


class KubernetesPodExPostOperator(KubernetesPodOperator):
    template_fields: Sequence[str] = (
        'image',
        'cmds',
        'arguments',
        'env_vars',
        'labels',
        'config_file',
        'pod_template_file',
        'namespace',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pre_execute(self, *args, **kwargs):
        set_default_volumn_mount(self, *args, **kwargs)
        # self.namespace = kwargs['context']['dag_run'].conf.get('ACCUTUNING_NAMESPACE')
        self.arguments = kwargs['context']['task_instance'].xcom_pull(
            task_ids='make_parameters', key='after_command').split()
        return super().pre_execute(*args, **kwargs)


class KubernetesPodExWorkerOperator(KubernetesPodOperator):
    template_fields: Sequence[str] = (
        'image',
        'cmds',
        'arguments',
        'env_vars',
        'labels',
        'config_file',
        'pod_template_file',
        'namespace',
    )
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pre_execute(self, *args, **kwargs):
        # self.namespace = kwargs['context']['dag_run'].conf.get('ACCUTUNING_NAMESPACE')
        set_default_volumn_mount(self, *args, **kwargs)
        return super().pre_execute(*args, **kwargs)


before_worker = KubernetesPodExPreOperator(
    namespace='default',
    image='{{dag_run.conf.ACCUTUNING_APP_IMAGE}}',
    # image='pooh97/accu-app:latest',
    # volumes=[volume],
    # volume_mounts=[volume_mount],
    name="before_worker",
    task_id="before_worker",
    env_vars=make_env_var(),
    cmds=["python3"],

    do_xcom_push=True,
    image_pull_policy='Always',
    get_logs=True,
    dag=dag,
)


worker_env = PythonOperator(task_id='make_worker_env', python_callable=make_worker_env, dag=dag)

worker = KubernetesPodExWorkerOperator(
    namespace=make_namespace(),
    image="{{dag_run.conf.ACCUTUNING_WORKER_IMAGE}}",
    name="worker",
    task_id="worker",
    env_vars={'ACCUTUNING_LOG_LEVEL': '{{dag_run.conf.ACCUTUNING_LOG_LEVEL}}',
              'ACCUTUNING_WORKSPACE': '{{ ti.xcom_pull(key="ACCUTUNING_WORKER_WORKSPACE") }}'},
    image_pull_policy='Always',
    get_logs=True,
    dag=dag,
)

worker_success = KubernetesPodExPostOperator(
    namespace=make_namespace(),
    image='{{dag_run.conf.ACCUTUNING_APP_IMAGE}}',
    name="worker_success",
    task_id="worker_success",
    env_vars=make_env_var(),
    # cmds=["python3"],
    # arguments=["/code/manage.py", ""{{dag_run.conf.ACCUTUNING_DJANGO_COMMAND']}}"", "--experiment={{dag_run.conf.ACCUTUNING_EXPERIMENT_ID']}}",  "--uuid={{dag_run.conf.ACCUTUNING_UUID']}}", "--timeout={{dag_run.conf.ACCUTUNING_TIMEOUT']}}"],
    cmds=["python3"],

    image_pull_policy='Always',
    get_logs=True,
    dag=dag,
    trigger_rule='all_success',
)

worker_fail = KubernetesPodExPostOperator(
    namespace=make_namespace(),
    image='{{dag_run.conf.ACCUTUNING_APP_IMAGE}}',
    name="worker_fail",
    task_id="worker_fail",
    env_vars=make_env_var(),
    # env_vars='{{dag_run.conf.worker_env_vars}}',
    # cmds=["python"],
    # arguments=["/code/manage.py", "ml_parse", "--experiment={{dag_run.conf.ACCUTUNING_EXPERIMENT_ID']}}",  "--uuid={{dag_run.conf.ACCUTUNING_UUID']}}", "--timeout={{dag_run.conf.ACCUTUNING_TIMEOUT']}}","--execute_range=after"],
    cmds=["python3"],
    image_pull_policy='Always',
    get_logs=True,
    dag=dag,
    trigger_rule='one_failed',
)


# one_success로 해야 skip된 task를 무시함
end = DummyOperator(
    task_id='end',
    trigger_rule='one_success',
    dag=dag,
)


start >> parameters >> before_worker >> worker_env >> worker

worker >> worker_success >> end
worker >> worker_fail >> end
