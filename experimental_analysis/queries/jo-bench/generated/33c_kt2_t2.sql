SELECT * FROM kind_type AS kt2, title AS t2 WHERE kt2.kind IN ('tv series', 'episode') AND t2.production_year BETWEEN 2000 AND 2010 AND kt2.id = t2.kind_id AND t2.kind_id = kt2.id;