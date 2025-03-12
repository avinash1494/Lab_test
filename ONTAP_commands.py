Report on NetApp ONTAP Commands and Volume Mounting Instructions
Commands to Explore the NetApp ONTAP
1. Command to show the cluster nodes:
aggr show
2. Command to create a snapshot:
snapshot create -vserver nvidia -volume dataneuron -snapshot test
3. Command to list all snapshots:
snapshot show -volume dataneuron
4. Command to create a FlexClone from the snapshot:
volume clone create Test -type RW -vserver nvidia -parent-volume dataneuron -parent-snapshot test
5. Command to list volumes:
volume show -vserver nvidia
6. Command to show the volumes and their list paths:
volume show -vserver nvidia -fields junction-path
7. Command to create a junction path for the volume:
volume mount -vserver nvidia -volume Test -junction-path /Test
8. Command to show the list of LIFs (Logical Interfaces):
network interface show -vserver nvidia
---
Steps for Mounting the Created Volume
1. Create a folder:
mkdir test
2. Mount the volume to the VM:
mount 10.195.60.223:/Test test
