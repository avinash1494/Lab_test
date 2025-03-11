Requested access has been provided. Please find the details below FYI!. Thank you.

 

[root@ci-bop-rau-01 ~]# ssh br38916@fas9000-60-53-55-CM.nb.openenglab.netapp.com

The authenticity of host 'fas9000-60-53-55-cm.nb.openenglab.netapp.com (10.195.60.57)' can't be established.

ED25519 key fingerprint is SHA256:vINkoRwyCErKTRlclaHfKfnvGGUbBZ3b216TwAafqt4.

This host key is known by the following other names/addresses:

    ~/.ssh/known_hosts:282: 10.195.60.57

Are you sure you want to continue connecting (yes/no/[fingerprint])? yes

Warning: Permanently added 'fas9000-60-53-55-cm.nb.openenglab.netapp.com' (ED25519) to the list of known hosts.

(br38916@fas9000-60-53-55-cm.nb.openenglab.netapp.com) Password:

 

Last login time: 2/28/2025 18:53:06

Unsuccessful login attempts since last login: 1

fas9000-60-53-55::> rows 0

 

fas9000-60-53-55::> cl sh

  (cluster show)

Node                  Health  Eligibility

--------------------- ------- ------------

fas9000-60-53-55-01   true    true

fas9000-60-53-55-02   true    true

2 entries were displayed.

 

fas9000-60-53-55::> system node show

Node      Health Eligibility Uptime        Model       Owner    Location

--------- ------ ----------- ------------- ----------- -------- ---------------

fas9000-60-53-55-01 true true 145 days 22:04 FAS9000            BLR

fas9000-60-53-55-02 true true 145 days 22:04 FAS9000

2 entries were displayed.

 

fas9000-60-53-55::> security login show -user-or-group-name br38916

 

Vserver: fas9000-60-53-55

                                                                 Second

User/Group                 Authentication                 Acct   Authentication

Name           Application Method        Role Name        Locked Method

-------------- ----------- ------------- ---------------- ------ --------------

br38916        http        password      admin            no     none

br38916        ontapi      password      admin            no     none

br38916        ssh         password      admin            no     none

3 entries were displayed.

 

fas9000-60-53-55::>

Y~r9jW!n5SAZJ*
