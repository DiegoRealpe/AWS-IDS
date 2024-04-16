import * as fs from 'fs';
import * as cdk from 'aws-cdk-lib';
import { Stack, StackProps, CfnOutput } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';

export class MyEC2Stack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // IAM resources
    // Create an IAM user
    const awsUser = iam.User.fromUserArn(this, 'AWS-Developer', "arn:aws:iam::844062109895:user/AWS-Developer");


    // EC2 resources
    // Create an SSH key pair
    const key = new ec2.KeyPair(this, 'EC2KeyPair', {
      type: ec2.KeyPairType.ED25519,
      keyPairName: 'ids_ec2_key', // Specify the name of the key pair
    });

    // Output the private key to a file
    const privateKeyPath = '/home/muzan/.ssh/ec2_private_key'; // Specify the path where you want to save the private key
    fs.writeFileSync(privateKeyPath, key.keyPairFingerprint); // Write the private key to a file
    console.log(`Private key saved to ${privateKeyPath}`);

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

    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      'yum update -y',
      'yum install -y python3',
      'pip install scikit-learn numpy matplotlib pandas scipy'
    )
    
    // EC2 Instance
    const instance = new ec2.Instance(this, 'IDS_EC2_Instance', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO), // Using t3.micro instance type
      machineImage: ubuntuAmi,
      vpc: vpc,
      keyPair: key,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      userData: userData,
      securityGroup: ec2InstanceSecurityGroup,
    });

    // Associate IAM user with EC2 instance
    instance.addToRolePolicy(new iam.PolicyStatement({
      actions: ['ssm:StartSession'],
      resources: [`arn:aws:iam::*:user/${awsUser.userName}`],
    }));

    // Output the public IP address to connect via SSH
    new CfnOutput(this, 'sshCommand', {
      value: `ssh ec2-user@${instance.instancePublicDnsName}`,
    });
  }
}

const app = new cdk.App();
new MyEC2Stack(app, 'MyEC2Stack');
