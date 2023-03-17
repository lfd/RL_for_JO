SELECT * FROM movie_keyword AS mk, keyword AS k, movie_info AS mi, aka_title AS at, info_type AS it1 WHERE it1.info = 'release dates' AND mi.note LIKE '%internet%' AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id AND mi.movie_id = at.movie_id AND at.movie_id = mi.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;