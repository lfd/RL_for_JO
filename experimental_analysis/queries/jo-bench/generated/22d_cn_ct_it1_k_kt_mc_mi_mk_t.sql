SELECT * FROM movie_keyword AS mk, info_type AS it1, kind_type AS kt, company_name AS cn, movie_companies AS mc, title AS t, company_type AS ct, movie_info AS mi, keyword AS k WHERE cn.country_code != '[us]' AND it1.info = 'countries' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND kt.kind IN ('movie', 'episode') AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Danish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id;