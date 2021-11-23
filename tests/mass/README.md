# Mass Testing

To run the mass tests:

```bash
docker-compose up --build -d
docker exec -it justuse-scraper bash

# To recreate the pypi.json cache:
python collect_packages.py

# To test the packages:
python justtest.py
```

## TODO

- Add in file size into the cache - currently tensorflow is at the top with stars and is huge...
- Tidy up code, document the intention and how users can add their own packages into the tests
- Add some unit tests to ensure the mass test functionality is working
