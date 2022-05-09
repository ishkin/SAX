# Process Mining Accelerator for Maximo

[comment]: <> (<details>)

[comment]: <> ( <summary>Table of Contents</summary>)

[comment]: <> (<!-- [Getting Started]&#40;#start&#41;<br>)

[comment]: <> ([Architecture]&#40;#architecture&#41;<br> -->)

[comment]: <> ([Setup]&#40;#Setup&#41;<br>)

[comment]: <> ([Running]&#40;#Running&#41;<br>)

[comment]: <> ([Configuration]&#40;#Configuration&#41;<br>)

[comment]: <> ([Architecture]&#40;#Architecture&#41;<br>)

[comment]: <> (</details>)


## Setup  
1. Download from zip or git clone the github project

2. Create the virtual python env and install the accelerator

`source install.sh`

3. Update the .env file in the home folder by copying your API keys and ENDPOINTs to the appropriate place 



## Running


Run the main python file
```
python main.py
```


# Advance

## Parameters
Run the main python file --help to see the options
```
python main.py --help
```

Options:
```
python main.py -limit 50                                        # limit the numbers of records to fetch        Defult value is 10000
python main.py -pagesize 50                                     # Set the page size to 50                      Defult value is 500
python main.py -config Predefined_config_files/WO_config.json   # Use your own configuration file              Defult is the config.json file in the home folder
python main.py -interactive                                     # Create a configuration file with user input  Defult value is None

```

```
python main.py -publish                                         # Publish the data set to IBM process miner                                      Defult value is Flase
python main.py -upload  my_file.csv                             # Upload a given data set to IBM process miner (path to csv file is required)    Defult value is Flase
python main.py -process_name Maximo_Asset_Management            # Set the process name                                                           Defult value is Maximo_Work_Management
python main.py -organization  my_org                            # Set the organization name                                                      Defult value is ""

```

## Data uploading to IPM Process Miner
Before running the accelerator please make sure to update the following fields in the .env file

`IPMAPIKEY  - this is the API key of IBM Process Miner`

`IPMUID     - this is the IBM Process Miner user ID (in demo-instance it is usually "maintenance.admin"`

`IPMAPIKEY  - this is the IBM Process miner base URL`
```
```

Please note: 

In order to upload the extracted data run ``python main.py -p``

In order to upload an existing CSV file run ``python main.py -u <path to CSV file>``


## The configuration file

Example: extracting the work order's status and their related asset and location
```
{
    "main_object": "wo",
    "api": "MXAPIWO",
    "status_to_extract_from_child": "wostatus",
    "relationships":
    {
      "ASSET": { "name": "asset" },
      "LOCATION": { "name": "location" }
    },
    "columns_to_extract": "*",
    "pm_mapping":
    {
      "pm_user": "changeby",
      "pm_action": "status",
      "pm_timestamp": "changedate",
      "pm_processid": "wonum"
    }
}

```

### Predefined configuration files

At the "Predefined config files" folder you may find different configuration file for different artifacts and relations

```

python main.py -config <path to config file>
```



### Configuration file explanation

1. main_object: the main artifact that should be explored; such as  wo, asset, sr and jp 

2. api : "MXAPI" + the name of the main object

3. status_to_extract_from_child: required when the status of the main object is in different table 

4. relationships:
	
	4.1 Add as many relationships as you like

	4.2 Make sure they exist

	4.3 The name of the relationship should be identical to the name appears in Maximo dataset

5. You may extract specific attributes by setting the "columns_to_extract" field. For example,
```
columns_to_extract :  [“status, wo.historyflag”, "wo.esttoolcost",“wo.asset.totdowntime”,"wo.asset.status_description", wo.location.type]
```

  5.1 Separate each attribute by a comma, and use prefix when needed

    5.1.1 if the attribute appears in the status table of the main object, just use the attribute name (e.g.,  status)

    5.1.2 if the attribute appears in the main object, use main_object.attribute (e.g.,  wo.wopriority)

    5.1.3 If the attribute appears in the related artifact, use mainobject.Related_artifact.attribute (e.g.,  wo.asset.totdowntime)

  5.2 Use * to extract all attributes


6. All fields are mandatory. Put "None" when necessary. For example,

```
{
  "main_object": "sr",
  "api": "MXAPISR",
  "status_to_extract_from_child": "tkstatus",
  "relationships": "None",
  "columns_to_extract": "*",
      "pm_mapping" : {
        "pm_timestamp": "changedate",
        "pm_processid": "ticketid",
        "pm_user": "changeby",
        "pm_action": "status"
      }
 }

```


## Interactive mode - creating a configuration file

A CMD-based UI that guides the user through use case and drive the accelerator automaitcly without edit config.json file

Run ```python main.py -interactive```

The configuration file will be saved at "Your_config_files" folder with the current timestamp

Note: you will be needed to choose:
1. Your main artifact
2. Whether you like to add a related artifact of interest
3. Whether to retrieve all attributes or specific one
  
     3.1 Separate each attribute by a comma

     3.2. Prefix related artifacts

     3.3. For example, wo.wopriority, wo.asset.totdowntime




