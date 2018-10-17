#!/usr/bin/python
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('in_file', nargs=1, type=argparse.FileType('r'))
    parser.add_argument('out_file', nargs='?', type=argparse.FileType('w'),
                        default='out.csv')
    in_file = parser.parse_args().in_file[0].read()
    out_file = parser.parse_args().out_file

    print('in_file = %s' % in_file)
    print('out_file = %s' % out_file)

    in_file = in_file.replace('","','###')    #To prevent deleting required commas
    in_file = in_file.replace(',','.')
    in_file = in_file.replace('###','","')

    out_file.write(in_file)


if __name__ == '__main__':
    main()
