# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  # For a complete reference, please see the online documentation at
  # https://docs.vagrantup.com.

  # Newest ubuntu box
  config.vm.box = "bento/ubuntu-16.04"

  # Setup folders
  config.vm.synced_folder "./src", "/home/vagrant/src"
  config.vm.synced_folder "./tests", "/home/vagrant/tests"

  # Provision
  config.vm.provision "shell", path: "./scripts/setup.sh"
  config.vm.provision "shell", path: "./scripts/install.sh"

end
