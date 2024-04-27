cd /home/ec2-user

yum update -y
yum groupinstall -y "Development Tools"
yum install -y amazon-cloudwatch-agent
yum install -y nc git zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel tk-devel libffi-devel xz-devel openssl11-devel

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -s -c ssm:us-east-2:AmazonCloudWatch-linux

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /home/ec2-user/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /home/ec2-user/.bashrc
echo 'eval "$(pyenv init -)"' >> /home/ec2-user/.bashrc

sudo -u ec2-user -i <<'EOF'

git clone https://github.com/pyenv/pyenv.git /home/ec2-user/.pyenv

export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pyenv install 3.11.0
pyenv global 3.11.0
pip install --upgrade pip
pip install scikit-learn numpy matplotlib pandas scipy
pip install git+https://github.com/hieulw/cicflowmeter

EOF
