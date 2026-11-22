"""Record the official frozen YAMNet cache without changing its weights."""
from soundsense.common import sha256, write_json
from .core import CACHE, REPORT, YAMNET_URL


def main():
    tfhub=CACHE/'tfhub'
    files={p.relative_to(tfhub).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size}
           for p in sorted(tfhub.rglob('*')) if p.is_file() and not p.name.endswith('.lock')}
    write_json(REPORT/'pretrained_artifacts.json',{'yamnet_transfer':{'source':YAMNET_URL,
          'version':1,'sample_rate':16000,'embedding_dimension':1024,'pooling':'mean over frames',
          'frozen':True,'extraction_device':'CPU','cached_files':files,
          'total_cache_bytes':sum(r['bytes'] for r in files.values())}})
    print('Official YAMNet artifact hashes saved',flush=True)


if __name__=='__main__':
    main()
