SELECT * FROM link_type AS lt, keyword AS k, movie_link AS ml, movie_keyword AS mk WHERE k.keyword IN ('sequel', 'revenge', 'based-on-novel') AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id;