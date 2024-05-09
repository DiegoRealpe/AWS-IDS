#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { MyEC2Stack } from '../lib/aws_ids-stack';

const app = new cdk.App();
new MyEC2Stack(app, 'MyEC2Stack', {});