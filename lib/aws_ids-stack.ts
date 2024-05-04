import * as fs from 'fs';
import * as cdk from 'aws-cdk-lib';
import { Stack, StackProps, CfnOutput } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';

export class MyEC2Stack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // IAM resources
    // Create an IAM user
    const awsUser = iam.User.fromUserArn(this, 'AWS-Developer', "arn:aws:iam::844062109895:user/AWS-Developer");

    // EC2 resources
    // Create an SSH key pair
    const key = ec2.KeyPair.fromKeyPairName(this, 'ec2loginkey', "ec2_login_key");

    // VPC
    const vpc = new ec2.Vpc(this, 'MyVpc', {
      maxAzs: 1, // For cost optimization, set to one AZ
      vpcName: "SimpleVPC",
      natGateways: 0,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: 'ServerPublic',
          subnetType: ec2.SubnetType.PUBLIC,
          mapPublicIpOnLaunch: true,
        },
      ]
    });

    // Ubuntu Amazon Machine Image (AMI) with Python and machine learning packages pre-installed
    const ubuntuAmi = new ec2.AmazonLinuxImage({
      generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2,
      edition: ec2.AmazonLinuxEdition.STANDARD,
      virtualization: ec2.AmazonLinuxVirt.HVM,
      storage: ec2.AmazonLinuxStorage.GENERAL_PURPOSE,
    });

    const ec2InstanceSecurityGroup = new ec2.SecurityGroup(
      this,
      'ec2InstanceSecurityGroup',
      { 
        vpc: vpc, 
        allowAllOutbound: true,
        description: 'Security Group for SSH'
      },
    );

    // Allow SSH inbound traffic on TCP port 22
    ec2InstanceSecurityGroup.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(22));
    
    const logGroup = new logs.LogGroup(this, 'EC2LogGroup', {
      removalPolicy: cdk.RemovalPolicy.DESTROY
    });
    
    const bucket = new s3.Bucket(this, 'MLBucket', {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Create an IAM role for EC2 instance
    const instanceRole = new iam.Role(this, 'EC2_Logging_Role', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("CloudWatchAgentServerPolicy")
      ]
    });
    logGroup.grantWrite(instanceRole)
    bucket.grantReadWrite(instanceRole)

    const cloudInit = ec2.CloudFormationInit.fromConfigSets({
      configSets: {
        default: ["install"],
      },
      configs: {
        install: new ec2.InitConfig([
          // Copy source files
          ec2.InitCommand.shellCommand(
            "echo 'export BUCKET_NAME=\"" + bucket.bucketName + "\"' >> /home/ec2-user/.bashrc"
          ),
          ec2.InitFile.fromFileInline(
            "/home/ec2-user/test_ids_model.py",
            "./src/test_ids_model.py",
          ),
          ec2.InitFile.fromFileInline(
            "/home/ec2-user/train_ids_model.py",
            "./src/train_ids_model.py",
          ),
          ec2.InitFile.fromFileInline(
            "/home/ec2-user/map_output_traffic.py",
            "./src/map_output_traffic.py",
          ),
          ec2.InitFile.fromFileInline(
            "/home/ec2-user/process_output_traffic.sh",
            "./src/process_output_traffic.sh",
          ),
          // Copy setup file
          ec2.InitFile.fromFileInline(
            "/etc/ec2_setup_script.sh",
            "./lib/ec2_setup_script.sh",
          ),
          ec2.InitCommand.shellCommand(
            "chmod +x /etc/ec2_setup_script.sh \
            /home/ec2-user/process_output_traffic.sh \
            /home/ec2-user/map_output_traffic.py \
            /home/ec2-user/train_ids_model.py \
            /home/ec2-user/test_ids_model.py"
          ),
          // // Copying some sample csv to test scripts
          ec2.InitFile.fromFileInline(
            "/home/ec2-user/dataset.csv",
            "./assets/dataset.csv",
          ),
          ec2.InitFile.fromFileInline(
            "/home/ec2-user/output.csv",
            "./assets/output.csv",
          ),
          ec2.InitCommand.shellCommand("chown -R ec2-user:ec2-user /home/ec2-user"),
          ec2.InitCommand.shellCommand("/etc/ec2_setup_script.sh"),
        ]),
      },
    })

    // EC2 Instance
    const instance = new ec2.Instance(this, "IDS_EC2_Instance", {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.SMALL), // Using t3.micro instance type
      // instanceType: ec2.InstanceType.of(ec2.InstanceClass.C6A, ec2.InstanceSize.XLARGE8), // Using t3.micro instance type
      machineImage: ubuntuAmi,
      vpc: vpc,
      keyPair: key,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      securityGroup: ec2InstanceSecurityGroup,
      role: instanceRole,
      init: cloudInit,
      initOptions: {
        timeout: cdk.Duration.minutes(15),
      },
    });

    // Associate IAM user with EC2 instance
    instance.addToRolePolicy(new iam.PolicyStatement({
      actions: ["ssm:StartSession"],
      resources: [`arn:aws:iam::*:user/${awsUser.userName}`],
    }));

    // Output the public IP address to connect via SSH
    new CfnOutput(this, "sshCommand", {
      value: `ssh ec2-user@${instance.instancePublicDnsName}`,
    });
    // Output the command to send a file to the bucket
    new CfnOutput(this, "bucket command", {
      value: `aws s3 cp "$FILE_NAME" "s3://${bucket.bucketName}/"`,
    });
  }
}

const app = new cdk.App();
new MyEC2Stack(app, "MyEC2Stack");
