SELECT * FROM keyword AS k, movie_keyword AS mk, movie_link AS ml, movie_info AS mi, movie_companies AS mc, company_type AS ct WHERE ct.kind = 'production companies' AND k.keyword = 'sequel' AND mc.note IS NULL AND mi.info IN ('Germany', 'German') AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;