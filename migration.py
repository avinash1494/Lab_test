import requests
import traceback
import urllib3
import json
import os
from requests.auth import HTTPBasicAuth

volumeDetails={}
volumeDetails["volumeName"]="volume_netapp_flex_1tb_root"
API_HOST="10.0.2.27"
USERNAME="fsxadmin"
PASSWORD="&N5dJ)WY4Xm4UqqV"


#@celery.task()
def create_snapshot_after_rag(workflow_id):
    try:
        VolumeName=volumeDetails["volumeName"]
        stage_info="vectorstore"
        snapshot_name = f"np{workflow_id}_{stage_info}"
        username = USERNAME
        password = PASSWORD
        print("calling API to get the NetAPP volume UUID !!!")
        url = f"https://{API_HOST}/api/storage/volumes?name={VolumeName}"
        try:
            response = requests.get(
            url,
            auth=HTTPBasicAuth(username, password),
            verify=False
            )
            if response.status_code == 200:
                result = response.json()
                records = result.get("records", [])
                print("records:",records)
            else:
                records=[]
        except requests.exceptions.RequestException as e:
            print(f"Error while getting Volume UUID: {e}")
            return {"status":False,"msg":e}
        if len(records)>0:
            volume_uuid=records[0]["uuid"]
        else:
            return {"status":False,"msg":"NO volume Found !!!"}
        print("calling API to create Snapshot!!!",volume_uuid)
        api_url =f"https://{API_HOST}/api/storage/volumes/{volume_uuid}/snapshots"
        headers = {
       "Content-Type": "application/json",
       "Accept": "application/json",
        }
        data = {"name": snapshot_name}
        try:
            response = requests.post(
                api_url,
                auth=(username, password),
                headers=headers,
                data=json.dumps(data),
                verify=False
            )
            snapshot_data = response.json()
            print("snapshot data:",snapshot_data)
            response.raise_for_status()
            print(f"Snapshot '{snapshot_name}' created successfully.")
            return {"status":True,"msg":f"Snapshot '{snapshot_name}' created successfully."}
        except requests.exceptions.RequestException as e:
            print(f"Error creating snapshot: {e}")
            return {"status":False,"msg":e}
    except Exception as e:
        error_msg={"task":"create_snapshot_after_rag","error":e,"traceback": traceback.format_exc()}
        print("error:",error_msg)
        return {"status":False,"msg":e}
# create_snapshot_after_rag("test1-23w2-1232")
#@celery.task()
def create_snapshot(workflow_id,stage_info):
    try:
        VolumeName=volumeDetails["volumeName"]
        snapshot_name = f"np{workflow_id}-{stage_info}"
        username = USERNAME
        password = PASSWORD
        print("calling API to get the NetAPP volume UUID !!!")
        url = f"https://{API_HOST}/api/storage/volumes?name={VolumeName}"
        try:
            response = requests.get(
            url,
            auth=HTTPBasicAuth(username, password),
            verify=False
            )
            if response.status_code == 200:
                result = response.json()
                records = result.get("records", [])
                print("records:",records)
            else:
                records=[]
        except requests.exceptions.RequestException as e:
            print(f"Error while getting Volume UUID: {e}")
            return {"status":False,"msg":e}
        if len(records)>0:
            volume_uuid=records[0]["uuid"]
        else:
            return {"status":False,"msg":"NO volume Found !!!"}
        
        print("calling API to create Snapshot!!!",volume_uuid)
        api_url =f"https://{API_HOST}/api/storage/volumes/{volume_uuid}/snapshots"
        headers = {
       "Content-Type": "application/json",
       "Accept": "application/json",
        }
        data = {"name": snapshot_name}
        try:
            response = requests.post(
                api_url,
                auth=(username, password),
                headers=headers,
                data=json.dumps(data),
                verify=False
            )
            response.raise_for_status()
            snapshot_data = response.json()
            print(f"Snapshot '{snapshot_name}' created successfully.")
            return {"status":True,"msg":f"Snapshot '{snapshot_name}' created successfully."}
        except requests.exceptions.RequestException as e:
            print(f"Error creating snapshot: {e}")
            return {"status":False,"msg":e}
    except Exception as e:
        error_msg={"task":"create_snapshot_after_rag","error":e,"traceback": traceback.format_exc()}
        print("error:",error_msg)
        return {"status":False,"msg":error_msg}
#create_snapshot("testing-snapshot-001","upload")

def list_snapshots(workflowId):
    try:
        VolumeName=volumeDetails["volumeName"]
        username = USERNAME
        password = PASSWORD
        print("calling API to get the NetAPP volume UUID !!!")
        url = f"https://{API_HOST}/api/storage/volumes?name={VolumeName}"
        try:
            response = requests.get(
            url,
            auth=HTTPBasicAuth(username, password),
            verify=False
            )
            if response.status_code == 200:
                result = response.json()
                records = result.get("records", [])
                print("records:",records)
            else:
                records=[]
        except requests.exceptions.RequestException as e:
            print(f"Error while getting Volume UUID: {e}")
            return {"status":False,"msg":e}
        if len(records)>0:
            volume_uuid=records[0]["uuid"]
        else:
            return {"status":False,"msg":"NO volume Found !!!"}
        print("calling API to list Snapshot!!!",volume_uuid)
        api_url =f"https://{API_HOST}/api/storage/volumes/{volume_uuid}/snapshots"
        headers = {
       "Content-Type": "application/json",
       "Accept": "application/json",
        }
        data = {}
        try:
            response = requests.get(
                api_url,
                auth=(username, password),
                headers=headers,
                verify=False
            )
            snapshot_data = response.json()
            print("snapshot data:",snapshot_data)
            response.raise_for_status()
            return {"status":True,"msg":f"Snapshot list retrieved successfully :\n'{snapshot_data}'"}
        except requests.exceptions.RequestException as e:
            print(f"Error creating snapshot: {e}")
            return {"status":False,"msg":e}
    except Exception as e:
        error_msg={"task":"list_snapshots","error":e,"traceback": traceback.format_exc()}
        print("error:",error_msg)
        return {"status":False,"msg":error_msg}
#list_snapshots("test")
    

def create_clone_from_existing_snapshot(workflow_id,parent_volume,new_volume_name, snapshot_name, share_name):
    try:
        print("calling API to create clone from the snapshot")
        username = USERNAME
        password = PASSWORD
        api_url =f"https://{API_HOST}/api/storage/volumes"
        headers = {
       "Content-Type": "application/json",
       "Accept": "application/json",
        }
        data={
            "name": new_volume_name,
            "svm": {
                "name": "ragsvm"
            },
            "clone": {
                "parent_volume": {
                    "name": parent_volume
                },
                "parent_snapshot": {
                    "name": snapshot_name
                },
                "parent_svm": {
                    "name": "ragsvm"
                }
            },
            "nas": {
                "path":f"/volume_netapp_flex_1tb/{new_volume_name}",
                "uid": 0,
                "gid": 0
            },
            "aggregates": [
                {
                    "name": "aggr1"
                }
            ],
            "comment": "Cloned volume for testing",
            "type": "rw"
        }
        # data = {"volume":new_volume_name,
        #         "parent-volume":parent_volume,
        #         "parent-vserver":"ragsvm",
        #         "parent-snapshot":snapshot_name,
        #         "junction-path":f"/volume_netapp_flex_1tb/{new_volume_name}",
        #         "qos-policy-group-name":"rag-pdfs-policy",
        #         "vserver":"ragsvm"}
        try:
            response = requests.post(
                api_url,
                auth=(username, password),
                headers=headers,
                data=json.dumps(data),
                verify=False
            )
            snapshot_data = response.json()
            print("snapshot_data:",snapshot_data)
            response.raise_for_status()
            return {"status":True,"msg":f"Snapshot list retrieved successfully :\n'{snapshot_data}'"}
        except requests.exceptions.RequestException as e:
            print(f"Error creating snapshot: {e}")
            return {"status":False,"msg":e}
        print("calling API to reterive the Volume UUID")
        url = f"https://{API_HOST}/api/storage/volumes?name={new_volume_name}"
        try:
            response = requests.get(
            url,
            auth=HTTPBasicAuth(username, password),
            verify=False
            )
            if response.status_code == 200:
                result = response.json()
                records = result.get("records", [])
                print("records:",records)
            else:
                records=[]
        except requests.exceptions.RequestException as e:
            print(f"Error while getting Volume UUID: {e}")
            return {"status":False,"msg":e}
        if len(records)>0:
            volume_uuid=records[0]["uuid"]
        else:
            return {"status":False,"msg":"NO volume Found !!!"}
        print("Found the new volume UUId:",volume_uuid)
        print("calling API to Modify the policy!!!!!")
        api_url =f"https://{API_HOST}/api/storage/volumes/{volume_uuid}"
        headers = {
       "Content-Type": "application/json",
       "Accept": "application/json",
        }
        data = {
          "nas": {
            "export_policy": {
              "name": "rag-pdfs-policy"
            }
          }
        }
        try:
            response = requests.put(
            api_url,
            auth=HTTPBasicAuth(username, password),
            verify=False
            )
            if response.status_code == 200:
                result = response.json()
                records = result.get("records", [])
                print("records:",records)
            else:
                records=[]
            #os.mkdir(f"/volume_netapp_flex_1tb/{new_volume_name}")
            os.system(f" sudo kubectl -n staging exec deploy/rag-celery -c volume-mounter -- mkdir -p /volume_netapp_flex_1tb/{new_volume_name}")
        except requests.exceptions.RequestException as e:
            print(f"Error while getting Volume UUID: {e}")
            return {"status":False,"msg":e}
    except Exception as e:
        error_msg={"task":"list_snapshots","error":e,"traceback": traceback.format_exc()}
        print("error:",error_msg)
        return {"status":False,"msg":error_msg}
        
create_clone_from_existing_snapshot("test","volume_netapp_flex_1tb_root","volume_netapp_flex_avinash_vol2", "nptesting-snapshot-001-upload", "")
